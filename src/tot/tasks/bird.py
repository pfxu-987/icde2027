import csv
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from typing import Any

from tot.prompts.bird import (
    decomp_repair_prompt,
    direct_sql_prompt,
    execution_value_prompt,
    propose_final_prompt,
    propose_step_prompt,
    value_prompt,
)
from tot.tasks.base import DATA_PATH, Task


_FORBIDDEN_SQL = re.compile(
    r"\b(?:insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum|reindex)\b",
    flags=re.IGNORECASE,
)


def _read_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"BIRD data file must contain a list: {path}")
        return [x for x in data if isinstance(x, dict)]

    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _extract_sql(text: str) -> str:
    if not text:
        return ""
    s = str(text).strip()

    fence = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", s, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        s = fence[-1].strip()

    m = re.search(r"\b(with|select)\b[\s\S]*", s, flags=re.IGNORECASE)
    if m:
        s = s[m.start() :].strip()

    # Keep the first complete statement. BIRD gold SQL is normally one SELECT.
    semi = s.find(";")
    if semi >= 0:
        s = s[: semi + 1]

    s = re.sub(r"^\s*SQL\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


def _is_safe_select(sql: str) -> bool:
    s = _extract_sql(sql)
    if not s:
        return False
    if _FORBIDDEN_SQL.search(s):
        return False
    return bool(re.match(r"^\s*(?:with|select)\b", s, flags=re.IGNORECASE))


def _normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return str(v)
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, int):
        return v
    text = str(v).strip()
    try:
        f = float(text)
        if f.is_integer():
            return int(f)
        return round(f, 6)
    except Exception:
        return text.lower()


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [tuple(_normalize_value(v) for v in row) for row in rows]


def _normalize_sql_key(sql: str) -> str:
    s = _extract_sql(sql).strip().rstrip(";")
    s = re.sub(r"--.*?$", " ", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([(),=<>+\-*/])\s*", r"\1", s)
    return s.lower().strip()


def _sql_clause_signature(sql: str) -> str:
    s = _normalize_sql_key(sql)
    if not s:
        return ""
    parts: list[str] = []
    markers = ["select", "from", "where", "group by", "having", "order by", "limit"]
    locs: list[tuple[int, str]] = []
    for m in markers:
        mm = re.search(rf"\b{re.escape(m)}\b", s)
        if mm:
            locs.append((mm.start(), m))
    locs.sort()
    for i, (pos, name) in enumerate(locs):
        end = locs[i + 1][0] if i + 1 < len(locs) else len(s)
        body = s[pos + len(name) : end].strip()
        if name in {"where", "group by"}:
            chunks = re.split(r"\band\b|,", body)
            body = "|".join(sorted(c.strip() for c in chunks if c.strip()))
        elif name == "from":
            # Alias names are often arbitrary; collapse repeated whitespace but keep join predicates.
            body = re.sub(r"\bas\s+\w+\b", "", body)
        parts.append(f"{name}:{body}")
    return " || ".join(parts)


def _rows_hash(rows: list[tuple[Any, ...]]) -> str:
    norm = _normalize_rows(rows)
    payload = {
        "n_rows": len(norm),
        "n_cols": len(norm[0]) if norm else 0,
        "rows_unordered": sorted([repr(r) for r in norm]),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _format_execution_feedback(valid: bool, rows: list[tuple[Any, ...]], error: str | None) -> str:
    if not valid:
        return f"SQL execution failed. Error: {error}"
    n_cols = len(rows[0]) if rows else 0
    return (
        "SQL execution succeeded. "
        f"Sampled rows: {len(rows)}. "
        f"Columns: {n_cols}. "
        f"Sample: {rows[:3]}"
    )


class BirdTask(Task):
    def __init__(
        self,
        file: str = "dev.json",
        db_root: str | None = None,
        steps: int = 5,
        max_schema_chars: int = 12000,
        max_result_rows: int = 2000,
    ):
        super().__init__()
        data_path = file if os.path.isabs(file) else os.path.join(DATA_PATH, "bird", file)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"BIRD data not found at {data_path}. Place dev.json/jsonl under src/tot/data/bird/ "
                "or pass --bird_file with an absolute path."
            )

        self.data = _read_json_or_jsonl(data_path)
        self.db_root = db_root or os.path.join(DATA_PATH, "bird", "dev_databases")
        self.steps = int(steps)
        self.max_schema_chars = int(max_schema_chars)
        self.max_result_rows = int(max_result_rows)
        self._schema_cache: dict[str, str] = {}
        self._exec_cache_lock = threading.Lock()
        self._exec_cache: dict[tuple[str, str], tuple[bool, tuple[tuple[Any, ...], ...], str | None]] = {}
        self._exec_cache_stats = {
            "calls": 0,
            "hits": 0,
            "misses": 0,
            "disabled_bypass": 0,
            "unsafe_or_empty_sql": 0,
            "errors": 0,
            "sql_timeouts": 0,
            "rows_truncated": 0,
            "elapsed_s": 0.0,
            "cost_sum": 0.0,
        }
        self.db_cost_tau_s = 1.0
        self.db_cost_row_weight = 0.2
        self.db_cost_cache_miss_penalty = 0.1
        self.sql_time_limit_s = float(os.environ.get("BIRD_SQL_EXEC_TIMEOUT_S", "30"))

        self.propose_stop = None
        self.propose_max_tokens = 512
        self.propose_temperature = 0.7

        self.value_stop = "\n"
        self.value_max_tokens = 16
        self.value_temperature = 0.0
        self.disable_cache = False

    def __len__(self) -> int:
        return len(self.data)

    def _db_id(self, idx: int) -> str:
        ex = self.data[idx]
        db_id = ex.get("db_id") or ex.get("database_id") or ex.get("db")
        if not db_id:
            raise KeyError(f"BIRD example {idx} has no db_id/database_id field")
        return str(db_id)

    def _db_path(self, db_id: str) -> str:
        candidates = [
            os.path.join(self.db_root, db_id, f"{db_id}.sqlite"),
            os.path.join(self.db_root, db_id, f"{db_id}.db"),
            os.path.join(self.db_root, db_id, "sqlite", f"{db_id}.sqlite"),
            os.path.join(self.db_root, db_id, "sqlite", f"{db_id}.db"),
            os.path.join(self.db_root, db_id, "sqlite", "database.sqlite"),
            os.path.join(self.db_root, db_id, "sqlite", "database.db"),
            os.path.join(self.db_root, db_id, "database.sqlite"),
            os.path.join(self.db_root, db_id, "database.db"),
            os.path.join(self.db_root, f"{db_id}.sqlite"),
            os.path.join(self.db_root, f"{db_id}.db"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"SQLite database for db_id={db_id!r} not found under {self.db_root}")

    def _description_dir(self, db_id: str) -> str | None:
        candidates = [
            os.path.join(self.db_root, db_id, "database_description"),
            os.path.join(self.db_root, "database_description", db_id),
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        return None

    def _load_column_descriptions(self, db_id: str) -> dict[tuple[str, str], str]:
        desc_dir = self._description_dir(db_id)
        if not desc_dir:
            return {}
        out: dict[tuple[str, str], str] = {}
        for name in os.listdir(desc_dir):
            if not name.lower().endswith(".csv"):
                continue
            table = os.path.splitext(name)[0]
            path = os.path.join(desc_dir, name)
            try:
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        keys = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
                        col = (
                            keys.get("column_name")
                            or keys.get("original_column_name")
                            or keys.get("column")
                            or keys.get("name")
                        )
                        desc = (
                            keys.get("column_description")
                            or keys.get("description")
                            or keys.get("value_description")
                            or keys.get("data_format")
                            or ""
                        )
                        if col and desc:
                            out[(table.lower(), str(col).lower())] = str(desc).strip()
            except Exception:
                continue
        return out

    def _schema_text(self, db_id: str) -> str:
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]

        db_path = self._db_path(db_id)
        descriptions = self._load_column_descriptions(db_id)
        lines: list[str] = []
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            tables = [r[0] for r in cur.fetchall()]

            for table in tables:
                cur.execute(f"PRAGMA table_info({self._quote_ident(table)})")
                cols = cur.fetchall()
                col_parts: list[str] = []
                for _, col_name, col_type, notnull, default, pk in cols:
                    part = f"{col_name} {col_type or 'TEXT'}"
                    if pk:
                        part += " primary key"
                    desc = descriptions.get((str(table).lower(), str(col_name).lower()))
                    if desc:
                        part += f" -- {desc}"
                    col_parts.append(part)
                lines.append(f"Table {table}:")
                lines.append("  columns: " + ", ".join(col_parts))

                cur.execute(f"PRAGMA foreign_key_list({self._quote_ident(table)})")
                fks = cur.fetchall()
                if fks:
                    fk_parts = [
                        f"{table}.{fk[3]} -> {fk[2]}.{fk[4]}"
                        for fk in fks
                        if fk[3] and fk[4]
                    ]
                    if not fk_parts:
                        continue
                    lines.append("  foreign keys: " + "; ".join(fk_parts))

        schema = "\n".join(lines)
        if len(schema) > self.max_schema_chars:
            schema = schema[: self.max_schema_chars] + "\n... [schema truncated]"
        self._schema_cache[db_id] = schema
        return schema

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    def get_input(self, idx: int) -> str:
        ex = self.data[idx]
        db_id = self._db_id(idx)
        question = ex.get("question") or ex.get("query") or ex.get("utterance") or ""
        evidence = ex.get("evidence") or ex.get("external_knowledge") or ex.get("knowledge") or ""
        difficulty = ex.get("difficulty") or ""

        parts = [
            f"Database id: {db_id}",
            f"Question: {question}",
        ]
        if evidence:
            parts.append(f"Evidence: {evidence}")
        if difficulty:
            parts.append(f"Difficulty: {difficulty}")
        parts.append("Schema:")
        parts.append(self._schema_text(db_id))
        return "\n".join(parts)

    def _gold_sql(self, idx: int) -> str:
        ex = self.data[idx]
        for key in ("SQL", "sql", "query", "gold_sql"):
            value = ex.get(key)
            if isinstance(value, str) and value.strip():
                return _extract_sql(value)
        return ""

    def _normalize_db_cost(self, elapsed_s: float, n_rows: int, cache_hit: bool) -> float:
        time_cost = min(1.0, max(0.0, float(elapsed_s) / max(1e-6, float(self.db_cost_tau_s))))
        row_cost = min(1.0, max(0.0, float(n_rows) / max(1.0, float(self.max_result_rows))))
        miss_cost = 0.0 if cache_hit else float(self.db_cost_cache_miss_penalty)
        return min(1.0, time_cost + float(self.db_cost_row_weight) * row_cost + miss_cost)

    def _execute_sql_with_meta(self, db_id: str, sql: str) -> tuple[bool, list[tuple[Any, ...]], str | None, dict[str, Any]]:
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
                "cache_hit": True,
                "elapsed_s": elapsed,
                "rows": 0,
                "db_cost": cost,
            }

        cache_disabled = bool(getattr(self, "disable_cache", False) or getattr(self, "disable_execution_cache", False))
        key = (str(db_id), _normalize_sql_key(query))
        with self._exec_cache_lock:
            self._exec_cache_stats["calls"] += 1
            if cache_disabled:
                self._exec_cache_stats["disabled_bypass"] += 1
            else:
                cached = self._exec_cache.get(key)
                if cached is not None:
                    self._exec_cache_stats["hits"] += 1
                    ok_cached, rows_cached, error_cached = cached
                    elapsed = time.perf_counter() - started
                    cost = self._normalize_db_cost(elapsed, len(rows_cached), True)
                    self._exec_cache_stats["elapsed_s"] += elapsed
                    self._exec_cache_stats["cost_sum"] += cost
                    return bool(ok_cached), list(rows_cached), error_cached, {
                        "cache_hit": True,
                        "elapsed_s": elapsed,
                        "rows": len(rows_cached),
                        "db_cost": cost,
                    }
            self._exec_cache_stats["misses"] += 1

        conn = None
        try:
            conn = sqlite3.connect(self._db_path(db_id), timeout=10.0)
            conn.execute("PRAGMA query_only = ON")
            sql_time_limit_s = float(getattr(self, "sql_time_limit_s", 0.0) or 0.0)
            if sql_time_limit_s > 0:
                deadline = started + sql_time_limit_s

                def _interrupt_if_timed_out() -> int:
                    return 1 if time.perf_counter() >= deadline else 0

                conn.set_progress_handler(_interrupt_if_timed_out, 10000)
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchmany(self.max_result_rows + 1)
            conn.set_progress_handler(None, 0)
            conn.close()
            if len(rows) > self.max_result_rows:
                rows = rows[: self.max_result_rows]
                with self._exec_cache_lock:
                    self._exec_cache_stats["rows_truncated"] += 1
            normalized = tuple(_normalize_rows(rows))
            elapsed = time.perf_counter() - started
            cost = self._normalize_db_cost(elapsed, len(normalized), False)
            with self._exec_cache_lock:
                if not cache_disabled:
                    self._exec_cache[key] = (True, normalized, None)
                self._exec_cache_stats["elapsed_s"] += elapsed
                self._exec_cache_stats["cost_sum"] += cost
            return True, list(normalized), None, {
                "cache_hit": False,
                "elapsed_s": elapsed,
                "rows": len(normalized),
                "db_cost": cost,
            }
        except Exception as e:
            if conn is not None:
                try:
                    conn.set_progress_handler(None, 0)
                    conn.close()
                except Exception:
                    pass
            error = str(e)
            elapsed = time.perf_counter() - started
            sql_time_limit_s = float(getattr(self, "sql_time_limit_s", 0.0) or 0.0)
            timed_out = sql_time_limit_s > 0 and elapsed >= sql_time_limit_s and "interrupted" in error.lower()
            if timed_out:
                error = f"sql_execution_timeout>{sql_time_limit_s:.1f}s"
            cost = self._normalize_db_cost(elapsed, 0, False)
            with self._exec_cache_lock:
                self._exec_cache_stats["errors"] += 1
                if timed_out:
                    self._exec_cache_stats["sql_timeouts"] += 1
                if not cache_disabled:
                    self._exec_cache[key] = (False, tuple(), error)
                self._exec_cache_stats["elapsed_s"] += elapsed
                self._exec_cache_stats["cost_sum"] += cost
            return False, [], error, {
                "cache_hit": False,
                "elapsed_s": elapsed,
                "rows": 0,
                "db_cost": cost,
            }

    def _execute_sql(self, db_id: str, sql: str) -> tuple[bool, list[tuple[Any, ...]], str | None]:
        ok, rows, error, _meta = self._execute_sql_with_meta(db_id, sql)
        return ok, rows, error

    def execution_cache_stats(self) -> dict[str, Any]:
        with self._exec_cache_lock:
            stats = dict(self._exec_cache_stats)
            stats["entries"] = len(self._exec_cache)
            calls = int(stats.get("calls") or 0)
            stats["hit_rate"] = float(stats.get("hits") or 0) / calls if calls else 0.0
            stats["avg_elapsed_s"] = float(stats.get("elapsed_s") or 0.0) / calls if calls else 0.0
            stats["avg_cost"] = float(stats.get("cost_sum") or 0.0) / calls if calls else 0.0
        return stats

    def test_output(self, idx: int, output: str):
        db_id = self._db_id(idx)
        pred_sql = _extract_sql(output)
        gold_sql = self._gold_sql(idx)
        if not pred_sql or not gold_sql:
            return {"r": 0, "pred": pred_sql, "gold": gold_sql, "error": "missing_sql"}

        gold_ok, gold_rows, gold_error = self._execute_sql(db_id, gold_sql)
        if not gold_ok:
            return {"r": 0, "pred": pred_sql, "gold": gold_sql, "error": f"gold_sql_error: {gold_error}"}

        pred_ok, pred_rows, pred_error = self._execute_sql(db_id, pred_sql)
        if not pred_ok:
            return {"r": 0, "pred": pred_sql, "gold": gold_sql, "error": pred_error}

        exact = pred_rows == gold_rows
        unordered = (
            sorted(pred_rows, key=repr) == sorted(gold_rows, key=repr)
            if len(pred_rows) == len(gold_rows)
            else False
        )
        return {
            "r": int(exact or unordered),
            "pred": pred_sql,
            "gold": gold_sql,
            "pred_rows": pred_rows[:20],
            "gold_rows": gold_rows[:20],
            "exact_order_match": bool(exact),
            "unordered_match": bool(unordered),
        }

    def propose_prompt_wrap(self, x: str, y: str = "", step: int | None = None) -> str:
        is_final = step is not None and int(step) >= int(self.steps) - 1
        if is_final:
            return propose_final_prompt.format(input=x, solution=y or "(none)")
        return propose_step_prompt.format(input=x, solution=y or "(none)")

    def parse_proposals(self, parent_y: str, response: str, step: int | None = None) -> list[str]:
        sql = _extract_sql(response)
        if not sql:
            return []
        if not sql.endswith(";"):
            sql += ";"
        return [sql]

    @staticmethod
    def dedup_proposals(proposals: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for p in proposals:
            sql = _extract_sql(p)
            key = re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(sql if sql.endswith(";") else sql + ";")
        return out

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return value_prompt.format(input=x, solution=_extract_sql(y) or y)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        scores: list[float] = []
        for output in value_outputs:
            raw = (output or "").strip()
            last = raw.splitlines()[-1].strip() if raw else ""
            m = re.fullmatch(r"-?\d+(?:\.\d+)?", last)
            if not m:
                m = re.search(r"score\s*[:：]?\s*(-?\d+(?:\.\d+)?)", raw, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"-?\d+(?:\.\d+)?", last or raw)
            if m:
                try:
                    scores.append(max(0.0, min(10.0, float(m.group(0 if m.lastindex is None else 1)))))
                    continue
                except Exception:
                    pass
            scores.append(5.0)
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def is_solved(self, idx: int, y: str) -> bool:
        return bool(self.test_output(idx, y).get("r"))


class BirdDecompRepairTask(BirdTask):
    """BIRD task variant for the existing layered/scheme_b schedulers.

    The scheduler is unchanged: it still calls propose/value and applies its
    own top-k, priority, batching, and concurrency policy. This task only
    changes the meaning of propose/value:
    - depth 0 proposes direct SQL seeds.
    - later depths repair existing SQL using execution feedback.
    - value combines execution validity with an LLM semantic score.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.propose_stop = None
        self.propose_max_tokens = 512
        self.propose_temperature = 0.7
        self.value_stop = "\n"
        self.value_max_tokens = 16
        self.value_temperature = 0.0
        self._semantic_cache_lock = threading.Lock()
        self._semantic_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _db_id_from_input(x: str) -> str:
        m = re.search(r"^Database id:\s*(.+?)\s*$", x or "", flags=re.MULTILINE)
        if not m:
            return ""
        return m.group(1).strip()

    def _feedback_for_input(self, x: str, sql: str) -> str:
        db_id = self._db_id_from_input(x)
        if not db_id:
            return "SQL execution was not run. Error: missing database id in prompt input."
        ok, rows, error = self._execute_sql(db_id, sql)
        return _format_execution_feedback(ok, rows, error)

    def _cache_for_input(self, x: str) -> dict[str, Any]:
        db_id = self._db_id_from_input(x) or "unknown_db"
        question = ""
        m = re.search(r"^Question:\s*(.+?)\s*$", x or "", flags=re.MULTILINE)
        if m:
            question = m.group(1).strip()
        key = f"{db_id}::{question}"
        with self._semantic_cache_lock:
            cache = self._semantic_cache.get(key)
            if cache is None:
                cache = {
                    "text": set(),
                    "structure": set(),
                    "structure_value": {},
                    "result": set(),
                    "stats": {
                        "seen": 0,
                        "text_duplicate": 0,
                        "structure_duplicate": 0,
                        "invalid_sql": 0,
                        "result_duplicate": 0,
                        "passed": 0,
                        "terminal": 0,
                    },
                }
                self._semantic_cache[key] = cache
            return cache

    def pre_value_score(self, x: str, y: str) -> dict[str, Any] | None:
        sql = _extract_sql(y)
        db_id = self._db_id_from_input(x)
        cache = self._cache_for_input(x)
        text_key = _normalize_sql_key(sql)
        structure_key = _sql_clause_signature(sql)
        cache_disabled = bool(getattr(self, "disable_cache", False) or getattr(self, "disable_semantic_cache", False))

        with self._semantic_cache_lock:
            cache["stats"]["seen"] += 1
            if not sql or not text_key:
                cache["stats"]["invalid_sql"] += 1
                cache["stats"]["terminal"] += 1
                return {
                    "skip": True,
                    "score": 0.0,
                    "reason": "empty_sql",
                    "terminal": True,
                    "expand": False,
                }
            if not cache_disabled and text_key in cache["text"]:
                cache["stats"]["text_duplicate"] += 1
                cache["stats"]["terminal"] += 1
                return {
                    "skip": True,
                    "score": 0.05,
                    "reason": "text_duplicate",
                    "terminal": True,
                    "expand": False,
                }
            if not cache_disabled and structure_key and structure_key in cache["structure"]:
                cache["stats"]["structure_duplicate"] += 1
                hit_value = float(cache.get("structure_value", {}).get(structure_key, 0.5) or 0.5)
                return {
                    "skip": True,
                    "score": max(0.1, 0.2 * hit_value),
                    "reason": "structure_duplicate_soft",
                    "terminal": False,
                    "expand": True,
                }

        if not db_id:
            return None

        ok, rows, _error, exec_meta = self._execute_sql_with_meta(db_id, sql)
        db_cost = float(exec_meta.get("db_cost") or 0.0)
        if not ok:
            with self._semantic_cache_lock:
                if not cache_disabled:
                    cache["text"].add(text_key)
                cache["stats"]["invalid_sql"] += 1
            return {
                "skip": True,
                "score": 0.0,
                "reason": "invalid_sql",
                "terminal": False,
                "expand": True,
                "db_penalty": db_cost,
                "db_cost": db_cost,
            }

        result_key = _rows_hash(rows)
        with self._semantic_cache_lock:
            if not cache_disabled and result_key in cache["result"]:
                cache["text"].add(text_key)
                if structure_key:
                    cache["structure"].add(structure_key)
                cache["stats"]["result_duplicate"] += 1
                return {
                    "skip": True,
                    "score": 0.15,
                    "reason": "result_duplicate",
                    "terminal": False,
                    "expand": True,
                    "db_penalty": db_cost,
                    "db_cost": db_cost,
                }
            if not cache_disabled:
                cache["text"].add(text_key)
                if structure_key:
                    cache["structure"].add(structure_key)
                cache["result"].add(result_key)
            cache["stats"]["passed"] += 1
        return {
            "skip": False,
            "reason": None,
            "terminal": False,
            "expand": True,
            "db_penalty": db_cost,
            "db_cost": db_cost,
        }

    def semantic_cache_stats(self) -> dict[str, Any]:
        merged = {
            "seen": 0,
            "text_duplicate": 0,
            "structure_duplicate": 0,
            "invalid_sql": 0,
            "result_duplicate": 0,
            "passed": 0,
            "terminal": 0,
            "n_questions": 0,
        }
        with self._semantic_cache_lock:
            merged["n_questions"] = len(self._semantic_cache)
            for cache in self._semantic_cache.values():
                for k, v in cache.get("stats", {}).items():
                    merged[k] = int(merged.get(k, 0)) + int(v)
        return merged

    def propose_prompt_wrap(self, x: str, y: str = "", step: int | None = None) -> str:
        sql = _extract_sql(y)
        if not sql:
            return direct_sql_prompt.format(input=x)
        feedback = self._feedback_for_input(x, sql)
        return decomp_repair_prompt.format(input=x, sql=sql, feedback=feedback)

    def parse_proposals(self, parent_y: str, response: str, step: int | None = None) -> list[str]:
        proposals = super().parse_proposals(parent_y, response, step)
        parent_key = _normalize_sql_key(parent_y)
        if not parent_key:
            return proposals
        return [p for p in proposals if _normalize_sql_key(p) != parent_key]

    def value_prompt_wrap(self, x: str, y: str) -> str:
        sql = _extract_sql(y)
        feedback = self._feedback_for_input(x, sql)
        return execution_value_prompt.format(input=x, sql=sql, feedback=feedback)

    def value_outputs_unwrap(self, x: str, y: str, value_outputs: list) -> float:
        sql = _extract_sql(y)
        db_id = self._db_id_from_input(x)
        if not sql or not db_id:
            return 0.0

        ok, rows, _error = self._execute_sql(db_id, sql)
        if not ok:
            return 0.0

        scores: list[float] = []
        for output in value_outputs:
            raw = (output or "").strip()
            last = raw.splitlines()[-1].strip() if raw else ""
            m = re.fullmatch(r"-?\d+(?:\.\d+)?", last)
            if not m:
                m = re.search(r"score\s*[:：]?\s*(-?\d+(?:\.\d+)?)", raw, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"-?\d+(?:\.\d+)?", last or raw)
            if m:
                try:
                    scores.append(max(0.0, min(10.0, float(m.group(0 if m.lastindex is None else 1)))))
                    continue
                except Exception:
                    pass
            scores.append(5.0)

        llm_score = float(sum(scores) / len(scores)) if scores else 0.0
        if rows:
            llm_score += 0.15
        structure_key = _sql_clause_signature(sql)
        if structure_key and not bool(getattr(self, "disable_cache", False) or getattr(self, "disable_semantic_cache", False)):
            cache = self._cache_for_input(x)
            with self._semantic_cache_lock:
                values = cache.setdefault("structure_value", {})
                old = float(values.get(structure_key, 0.0) or 0.0)
                values[structure_key] = max(old, float(llm_score))
        return float(llm_score)
