import os
import sqlite3
import time
from typing import Any

from tot.prompts.spider import decomp_repair_prompt, direct_sql_prompt, execution_value_prompt
from tot.tasks.base import DATA_PATH
from tot.tasks.bird import (
    _extract_sql,
    _format_execution_feedback,
    _is_safe_select,
    _normalize_rows,
    _normalize_sql_key,
)
from tot.tasks.bird_plan_proxy import BirdPlanProxyTask


def _resolve_spider_file(file: str) -> str:
    return file if os.path.isabs(file) else os.path.join(DATA_PATH, "spider", file)


def _resolve_spider_db_root(db_root: str | None) -> str:
    if db_root:
        return db_root
    return os.path.join(DATA_PATH, "spider", "database")


class SpiderPlanProxyTask(BirdPlanProxyTask):
    """Spider decomp-repair task aligned with BirdPlanProxyTask.

    This keeps the new BIRD value/cache logic:
    - structure duplicates are soft-scored and can still be repaired.
    - value results carry db_penalty/db_cost into the scheduler priority.
    - EXPLAIN QUERY PLAN provides a calibrated physical proxy penalty.

    Only Spider-specific data loading, input formatting, gold SQL extraction,
    and prompt wording are overridden.
    """

    def __init__(
        self,
        file: str = "dev.json",
        db_root: str | None = None,
        steps: int = 5,
        max_schema_chars: int = 12000,
        max_result_rows: int = 2000,
        disable_semantic_cache: bool = False,
        disable_execution_cache: bool = False,
    ):
        data_path = _resolve_spider_file(file)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Spider data not found at {data_path}. Place dev.json under src/tot/data/spider/ "
                "or pass --spider_file with an absolute path."
            )
        super().__init__(
            file=data_path,
            db_root=_resolve_spider_db_root(db_root),
            steps=steps,
            max_schema_chars=max_schema_chars,
            max_result_rows=max_result_rows,
        )
        self.disable_semantic_cache = bool(disable_semantic_cache)
        self.disable_execution_cache = bool(disable_execution_cache)

    def _execute_sql_with_meta(self, db_id: str, sql: str) -> tuple[bool, list[tuple[Any, ...]], str | None, dict[str, Any]]:
        if not bool(getattr(self, "disable_execution_cache", False)):
            return super()._execute_sql_with_meta(db_id, sql)

        started = time.perf_counter()
        query = _extract_sql(sql)
        if not _is_safe_select(query):
            elapsed = time.perf_counter() - started
            cost = self._normalize_db_cost(elapsed, 0, True)
            with self._exec_cache_lock:
                self._exec_cache_stats["calls"] += 1
                self._exec_cache_stats["unsafe_or_empty_sql"] += 1
                self._exec_cache_stats["elapsed_s"] += elapsed
                self._exec_cache_stats["cost_sum"] += cost
            return False, [], "unsafe_or_empty_sql", {
                "cache_hit": False,
                "elapsed_s": elapsed,
                "rows": 0,
                "db_cost": cost,
            }

        with self._exec_cache_lock:
            self._exec_cache_stats["calls"] += 1
            self._exec_cache_stats["misses"] += 1

        try:
            conn = sqlite3.connect(self._db_path(db_id), timeout=10.0)
            conn.execute("PRAGMA query_only = ON")
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchmany(self.max_result_rows + 1)
            conn.close()
            if len(rows) > self.max_result_rows:
                rows = rows[: self.max_result_rows]
                with self._exec_cache_lock:
                    self._exec_cache_stats["rows_truncated"] += 1
            normalized = _normalize_rows(rows)
            elapsed = time.perf_counter() - started
            cost = self._normalize_db_cost(elapsed, len(normalized), False)
            with self._exec_cache_lock:
                self._exec_cache_stats["elapsed_s"] += elapsed
                self._exec_cache_stats["cost_sum"] += cost
            return True, normalized, None, {
                "cache_hit": False,
                "elapsed_s": elapsed,
                "rows": len(normalized),
                "db_cost": cost,
            }
        except Exception as e:
            elapsed = time.perf_counter() - started
            cost = self._normalize_db_cost(elapsed, 0, False)
            with self._exec_cache_lock:
                self._exec_cache_stats["errors"] += 1
                self._exec_cache_stats["elapsed_s"] += elapsed
                self._exec_cache_stats["cost_sum"] += cost
            return False, [], str(e), {
                "cache_hit": False,
                "elapsed_s": elapsed,
                "rows": 0,
                "db_cost": cost,
            }

    def pre_value_score(self, x: str, y: str) -> dict[str, Any] | None:
        if not bool(getattr(self, "disable_semantic_cache", False)):
            return super().pre_value_score(x, y)

        # Keep the plan-proxy value formula, but do not read/write semantic
        # duplicate/result caches. This isolates semantic cache contribution.
        sql = _extract_sql(y)
        db_id = self._db_id_from_input(x)
        if not sql or not _normalize_sql_key(sql):
            return {
                "skip": True,
                "score": 0.0,
                "reason": "empty_sql",
                "terminal": True,
                "expand": False,
                "db_penalty": 1.0,
                "db_cost": 1.0,
            }
        if not db_id:
            return None

        proxy_ok, proxy_cost, proxy_meta = self._proxy_cost(db_id, sql)
        if not proxy_ok:
            with self._proxy_lock:
                self._proxy_stats["explain_failures"] += 1
            self._record_proxy_cost(db_id, 1.0, 1.0, proxy_meta, hard_pruned=True)
            return {
                "skip": True,
                "score": 0.0,
                "reason": proxy_meta.get("reason") or "explain_failure",
                "terminal": True,
                "expand": False,
                "db_penalty": 1.0,
                "db_cost": 1.0,
            }

        b_proxy = self._proxy_bmax(db_id, proxy_cost)
        proxy_norm = min(1.0, float(proxy_cost) / b_proxy)
        self._record_proxy_cost(db_id, proxy_cost, proxy_norm, proxy_meta)

        ok, _rows, _error, _exec_meta = self._execute_sql_with_meta(db_id, sql)
        if not ok:
            return {
                "skip": True,
                "score": 0.0,
                "reason": "invalid_sql",
                "terminal": False,
                "expand": True,
                "db_penalty": proxy_norm,
                "db_cost": proxy_norm,
            }

        return {
            "skip": False,
            "reason": None,
            "terminal": False,
            "expand": True,
            "db_penalty": proxy_norm,
            "db_cost": proxy_norm,
        }

    def semantic_cache_stats(self) -> dict[str, Any]:
        stats = super().semantic_cache_stats()
        stats["disabled"] = bool(getattr(self, "disable_semantic_cache", False))
        return stats

    def _gold_sql(self, idx: int) -> str:
        ex = self.data[idx]
        for key in ("query", "SQL", "sql", "gold_sql"):
            value = ex.get(key)
            if isinstance(value, str) and value.strip():
                return _extract_sql(value)
        return ""

    def get_input(self, idx: int) -> str:
        ex = self.data[idx]
        db_id = self._db_id(idx)
        question = ex.get("question") or ex.get("utterance") or ""
        return "\n".join(
            [
                f"Database id: {db_id}",
                f"Question: {question}",
                "Schema:",
                self._schema_text(db_id),
            ]
        )

    def propose_prompt_wrap(self, x: str, y: str = "", step: int | None = None) -> str:
        sql = _extract_sql(y)
        if not sql:
            return direct_sql_prompt.format(input=x)
        feedback = self._feedback_for_input(x, sql)
        return decomp_repair_prompt.format(input=x, sql=sql, feedback=feedback)

    def value_prompt_wrap(self, x: str, y: str) -> str:
        sql = _extract_sql(y)
        feedback = self._feedback_for_input(x, sql)
        return execution_value_prompt.format(input=x, sql=sql, feedback=feedback)

    def _feedback_for_input(self, x: str, sql: str) -> str:
        db_id = self._db_id_from_input(x)
        if not db_id:
            return "SQL execution was not run. Error: missing database id in prompt input."
        ok, rows, error = self._execute_sql(db_id, sql)
        return _format_execution_feedback(ok, rows, error)

    @staticmethod
    def dedup_proposals(proposals: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for proposal in proposals:
            key = _normalize_sql_key(proposal)
            if not key or key in seen:
                continue
            seen.add(key)
            sql = _extract_sql(proposal)
            out.append(sql if sql.endswith(";") else sql + ";")
        return out
