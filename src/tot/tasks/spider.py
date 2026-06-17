import os
import sqlite3
from typing import Any

from tot.prompts.spider import (
    decomp_repair_prompt,
    direct_sql_prompt,
    execution_value_prompt,
    propose_final_prompt,
    propose_step_prompt,
    value_prompt,
)
from tot.tasks.base import DATA_PATH
from tot.tasks.bird import (
    BirdDecompRepairTask,
    BirdTask,
    _extract_sql,
    _format_execution_feedback,
    _is_safe_select,
    _normalize_sql_key,
    _normalize_rows,
)


def _resolve_spider_file(file: str) -> str:
    return file if os.path.isabs(file) else os.path.join(DATA_PATH, "spider", file)


def _resolve_spider_db_root(db_root: str | None) -> str:
    if db_root:
        return db_root
    return os.path.join(DATA_PATH, "spider", "database")


class SpiderTask(BirdTask):
    """Spider text-to-SQL task using the existing ToT schedulers.

    The data format follows the official Spider release:
    - dev.json / train_spider.json contains question, query, db_id
    - database/<db_id>/<db_id>.sqlite contains the SQLite database

    Evaluation is execution-result matching, consistent with the BIRD adapter.
    This is useful for runnable system experiments, but it is not a replacement
    for Spider's official exact-match/evaluator script.
    """

    def __init__(
        self,
        file: str = "dev.json",
        db_root: str | None = None,
        steps: int = 5,
        max_schema_chars: int = 12000,
        max_result_rows: int = 2000,
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
        self.disable_execution_cache = bool(disable_execution_cache)

    def _execute_sql(self, db_id: str, sql: str) -> tuple[bool, list[tuple[Any, ...]], str | None]:
        if not bool(getattr(self, "disable_execution_cache", False)):
            return super()._execute_sql(db_id, sql)

        query = _extract_sql(sql)
        if not _is_safe_select(query):
            with self._exec_cache_lock:
                self._exec_cache_stats["calls"] += 1
                self._exec_cache_stats["unsafe_or_empty_sql"] += 1
            return False, [], "unsafe_or_empty_sql"

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
            return True, _normalize_rows(rows), None
        except Exception as e:
            with self._exec_cache_lock:
                self._exec_cache_stats["errors"] += 1
            return False, [], str(e)

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
        is_final = step is not None and int(step) >= int(self.steps) - 1
        if is_final:
            return propose_final_prompt.format(input=x, solution=y or "(none)")
        return propose_step_prompt.format(input=x, solution=y or "(none)")

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return value_prompt.format(input=x, solution=_extract_sql(y) or y)


class SpiderDecompRepairTask(BirdDecompRepairTask, SpiderTask):
    """Spider variant of the BIRD decompose-and-repair search semantics."""

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
            disable_execution_cache=disable_execution_cache,
        )
        self.disable_semantic_cache = bool(disable_semantic_cache)

    def pre_value_score(self, x: str, y: str) -> dict[str, Any] | None:
        if bool(getattr(self, "disable_semantic_cache", False)):
            return None
        return super().pre_value_score(x, y)

    def semantic_cache_stats(self) -> dict[str, Any]:
        stats = super().semantic_cache_stats()
        stats["disabled"] = bool(getattr(self, "disable_semantic_cache", False))
        return stats

    def get_input(self, idx: int) -> str:
        return SpiderTask.get_input(self, idx)

    def _gold_sql(self, idx: int) -> str:
        return SpiderTask._gold_sql(self, idx)

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
