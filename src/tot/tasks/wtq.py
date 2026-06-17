import csv
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from typing import Any

from tot.prompts.wtq import direct_sql_prompt, decomp_repair_prompt, execution_value_prompt
from tot.tasks.base import DATA_PATH, Task
from tot.tasks.bird import (
    _extract_sql,
    _format_execution_feedback,
    _is_safe_select,
    _normalize_sql_key,
    _rows_hash,
    _sql_clause_signature,
)


def _read_tsv(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _safe_col(name: str, used: set[str]) -> str:
    base = re.sub(r"\W+", "_", str(name).strip().lower()).strip("_") or "col"
    if re.match(r"^\d", base):
        base = "c_" + base
    out = base
    i = 2
    while out in used:
        out = f"{base}_{i}"
        i += 1
    used.add(out)
    return out


def _normalize_answer_value(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).strip()
    text = re.sub(r"^\s*(?:\\\"|\"|')|(?:\\\"|\"|')\s*$", "", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"^\$|%$", "", text)
    text = text.replace(",", "")
    try:
        f = float(text)
        if f.is_integer():
            return str(int(f))
        return f"{f:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return text


def _parse_target_values(raw: str) -> list[str]:
    s = str(raw or "").strip()
    if not s:
        return []
    quoted = re.findall(r'\(description\s+"([^"]*)"\)', s)
    if quoted:
        return [_normalize_answer_value(x) for x in quoted if _normalize_answer_value(x)]
    quoted = re.findall(r'"([^"]*)"', s)
    if quoted:
        return [_normalize_answer_value(x) for x in quoted if _normalize_answer_value(x)]
    if "|" in s:
        return [_normalize_answer_value(x) for x in s.split("|") if _normalize_answer_value(x)]
    if "\t" in s:
        return [_normalize_answer_value(x) for x in s.split("\t") if _normalize_answer_value(x)]
    return [_normalize_answer_value(s)]


def _flatten_rows(rows: list[tuple[Any, ...]]) -> list[str]:
    out: list[str] = []
    for row in rows:
        for v in row:
            norm = _normalize_answer_value(v)
            if norm:
                out.append(norm)
    return out


class WTQTask(Task):
    def __init__(
        self,
        file: str = "WikiTableQuestions/data/random-split-1-dev.tsv",
        data_root: str | None = None,
        steps: int = 5,
        max_schema_chars: int = 12000,
        max_result_rows: int = 2000,
    ):
        super().__init__()
        self.data_root = data_root or os.path.join(DATA_PATH, "wtq")
        data_path = file if os.path.isabs(file) else os.path.join(self.data_root, file)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"WTQ data file not found at {data_path}. "
                "Download WikiTableQuestions under src/tot/data/wtq/ or pass --wtq_file."
            )
        self.data_path = data_path
        self.data = _read_tsv(data_path)
        self.steps = int(steps)
        self.max_schema_chars = int(max_schema_chars)
        self.max_result_rows = int(max_result_rows)
        self.sqlite_cache_dir = os.path.join(self.data_root, "sqlite_cache")
        os.makedirs(self.sqlite_cache_dir, exist_ok=True)
        self._table_meta_lock = threading.Lock()
        self._table_meta_cache: dict[str, dict[str, Any]] = {}
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
            "rows_truncated": 0,
            "elapsed_s": 0.0,
            "cost_sum": 0.0,
        }
        self.db_cost_tau_s = 0.5
        self.db_cost_row_weight = 0.2
        self.db_cost_cache_miss_penalty = 0.1
        self.disable_cache = False

        self.propose_stop = None
        self.propose_max_tokens = 512
        self.propose_temperature = 0.7
        self.value_stop = "\n"
        self.value_max_tokens = 16
        self.value_temperature = 0.0

    def __len__(self) -> int:
        return len(self.data)

    def _question(self, idx: int) -> str:
        ex = self.data[idx]
        return ex.get("utterance") or ex.get("question") or ex.get("Question") or ""

    def _target_values(self, idx: int) -> list[str]:
        ex = self.data[idx]
        raw = ex.get("targetValue") or ex.get("answer") or ex.get("answers") or ""
        return _parse_target_values(raw)

    def _table_name(self, idx: int) -> str:
        ex = self.data[idx]
        table = ex.get("context") or ex.get("table") or ex.get("table_file") or ex.get("table_name") or ""
        table = str(table).strip()
        if table.startswith("csv/") or table.endswith(".csv") or table.endswith(".tsv"):
            return table
        raise KeyError(f"WTQ example {idx} has no table context field")

    def _table_path(self, table_name: str) -> str:
        candidates = [
            os.path.join(self.data_root, "WikiTableQuestions", table_name),
            os.path.join(self.data_root, table_name),
            os.path.join(os.path.dirname(self.data_path), table_name),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"WTQ table {table_name!r} not found under {self.data_root}")

    def _read_table(self, table_path: str) -> tuple[list[str], list[list[str]]]:
        with open(table_path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            except Exception:
                dialect = csv.excel
            rows = list(csv.reader(f, dialect))
        if not rows:
            return [], []
        header = [str(x).strip() or f"column_{i+1}" for i, x in enumerate(rows[0])]
        body = [[str(x).strip() for x in r] for r in rows[1:]]
        return header, body

    def _table_meta(self, table_name: str) -> dict[str, Any]:
        with self._table_meta_lock:
            cached = self._table_meta_cache.get(table_name)
            if cached is not None:
                return cached

        path = self._table_path(table_name)
        header, rows = self._read_table(path)
        used: set[str] = set()
        col_map = [(orig, _safe_col(orig, used)) for orig in header]
        digest = hashlib.sha1(table_name.encode("utf-8")).hexdigest()[:16]
        sqlite_path = os.path.join(self.sqlite_cache_dir, f"{digest}.sqlite")
        if not os.path.exists(sqlite_path):
            with sqlite3.connect(sqlite_path) as conn:
                cur = conn.cursor()
                cur.execute('DROP TABLE IF EXISTS "w"')
                cols_sql = ", ".join(f'"{safe}" TEXT' for _orig, safe in col_map)
                cur.execute(f'CREATE TABLE "w" ({cols_sql})')
                if col_map and rows:
                    placeholders = ", ".join(["?"] * len(col_map))
                    col_names = ", ".join(f'"{safe}"' for _orig, safe in col_map)
                    fixed_rows = [(r + [""] * len(col_map))[: len(col_map)] for r in rows]
                    cur.executemany(f'INSERT INTO "w" ({col_names}) VALUES ({placeholders})', fixed_rows)
                conn.commit()
        meta = {
            "table_name": table_name,
            "table_path": path,
            "sqlite_path": sqlite_path,
            "columns": col_map,
            "n_rows": len(rows),
            "sample_rows": rows[:3],
        }
        with self._table_meta_lock:
            self._table_meta_cache[table_name] = meta
        return meta

    def _db_id(self, idx: int) -> str:
        return self._table_name(idx)

    def _db_path(self, db_id: str) -> str:
        return str(self._table_meta(db_id)["sqlite_path"])

    def _schema_text(self, table_name: str) -> str:
        if table_name in self._schema_cache:
            return self._schema_cache[table_name]
        meta = self._table_meta(table_name)
        lines = ["Table w:"]
        lines.append("  columns:")
        for orig, safe in meta["columns"]:
            lines.append(f"    {safe} TEXT -- original header: {orig}")
        lines.append(f"  rows: {meta['n_rows']}")
        if meta["sample_rows"]:
            lines.append("  sample rows:")
            for row in meta["sample_rows"]:
                row_obj = {safe: (row[i] if i < len(row) else "") for i, (_orig, safe) in enumerate(meta["columns"])}
                lines.append("    " + json.dumps(row_obj, ensure_ascii=False))
        schema = "\n".join(lines)
        if len(schema) > self.max_schema_chars:
            schema = schema[: self.max_schema_chars] + "\n... [schema truncated]"
        self._schema_cache[table_name] = schema
        return schema

    def get_input(self, idx: int) -> str:
        table_name = self._table_name(idx)
        parts = [
            f"Table id: {table_name}",
            f"Question: {self._question(idx)}",
            "Schema:",
            self._schema_text(table_name),
        ]
        return "\n".join(parts)

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
            return False, [], "unsafe_or_empty_sql", {"cache_hit": True, "elapsed_s": elapsed, "rows": 0, "db_cost": cost}

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

        try:
            with sqlite3.connect(self._db_path(db_id), timeout=10.0) as conn:
                conn.execute("PRAGMA query_only = ON")
                cur = conn.cursor()
                cur.execute(query)
                rows = cur.fetchmany(self.max_result_rows + 1)
            if len(rows) > self.max_result_rows:
                rows = rows[: self.max_result_rows]
                with self._exec_cache_lock:
                    self._exec_cache_stats["rows_truncated"] += 1
            normalized = tuple(tuple(str(v) if v is not None else "" for v in row) for row in rows)
            elapsed = time.perf_counter() - started
            cost = self._normalize_db_cost(elapsed, len(normalized), False)
            with self._exec_cache_lock:
                if not cache_disabled:
                    self._exec_cache[key] = (True, normalized, None)
                self._exec_cache_stats["elapsed_s"] += elapsed
                self._exec_cache_stats["cost_sum"] += cost
            return True, list(normalized), None, {"cache_hit": False, "elapsed_s": elapsed, "rows": len(normalized), "db_cost": cost}
        except Exception as e:
            error = str(e)
            elapsed = time.perf_counter() - started
            cost = self._normalize_db_cost(elapsed, 0, False)
            with self._exec_cache_lock:
                self._exec_cache_stats["errors"] += 1
                if not cache_disabled:
                    self._exec_cache[key] = (False, tuple(), error)
                self._exec_cache_stats["elapsed_s"] += elapsed
                self._exec_cache_stats["cost_sum"] += cost
            return False, [], error, {"cache_hit": False, "elapsed_s": elapsed, "rows": 0, "db_cost": cost}

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
        gold = self._target_values(idx)
        if not pred_sql:
            return {"r": 0, "pred": pred_sql, "gold": gold, "error": "missing_sql"}
        pred_ok, pred_rows, pred_error = self._execute_sql(db_id, pred_sql)
        if not pred_ok:
            return {"r": 0, "pred": pred_sql, "gold": gold, "error": pred_error}
        pred_values = _flatten_rows(pred_rows)
        pred_set = sorted(pred_values)
        gold_set = sorted(gold)
        exact = pred_set == gold_set
        contains = bool(gold_set) and all(g in pred_values for g in gold_set) and len(pred_values) <= max(len(gold_set) + 2, 5)
        return {
            "r": int(exact or contains),
            "pred": pred_sql,
            "gold": gold,
            "pred_values": pred_values[:30],
            "exact_denotation_match": bool(exact),
            "contains_gold_with_small_extra": bool(contains),
        }

    def is_solved(self, idx: int, y: str) -> bool:
        return bool(self.test_output(idx, y).get("r"))


class WTQDecompRepairTask(WTQTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._semantic_cache_lock = threading.Lock()
        self._semantic_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _db_id_from_input(x: str) -> str:
        m = re.search(r"^Table id:\s*(.+?)\s*$", x or "", flags=re.MULTILINE)
        return m.group(1).strip() if m else ""

    def _feedback_for_input(self, x: str, sql: str) -> str:
        db_id = self._db_id_from_input(x)
        if not db_id:
            return "SQL execution was not run. Error: missing table id in prompt input."
        ok, rows, error = self._execute_sql(db_id, sql)
        return _format_execution_feedback(ok, rows, error)

    def _cache_for_input(self, x: str) -> dict[str, Any]:
        db_id = self._db_id_from_input(x) or "unknown_table"
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
                return {"skip": True, "score": 0.0, "reason": "empty_sql", "terminal": True, "expand": False}
            if not cache_disabled and text_key in cache["text"]:
                cache["stats"]["text_duplicate"] += 1
                cache["stats"]["terminal"] += 1
                return {"skip": True, "score": 0.05, "reason": "text_duplicate", "terminal": True, "expand": False}
            if not cache_disabled and structure_key and structure_key in cache["structure"]:
                cache["stats"]["structure_duplicate"] += 1
                hit_value = float(cache.get("structure_value", {}).get(structure_key, 0.5) or 0.5)
                return {"skip": True, "score": max(0.1, 0.2 * hit_value), "reason": "structure_duplicate_soft", "terminal": False, "expand": True}
        if not db_id:
            return None
        ok, rows, _error, exec_meta = self._execute_sql_with_meta(db_id, sql)
        db_cost = float(exec_meta.get("db_cost") or 0.0)
        if not ok:
            with self._semantic_cache_lock:
                if not cache_disabled:
                    cache["text"].add(text_key)
                cache["stats"]["invalid_sql"] += 1
            return {"skip": True, "score": 0.0, "reason": "invalid_sql", "terminal": False, "expand": True, "db_penalty": db_cost, "db_cost": db_cost}
        result_key = _rows_hash(rows)
        with self._semantic_cache_lock:
            if not cache_disabled and result_key in cache["result"]:
                cache["text"].add(text_key)
                if structure_key:
                    cache["structure"].add(structure_key)
                cache["stats"]["result_duplicate"] += 1
                return {"skip": True, "score": 0.15, "reason": "result_duplicate", "terminal": False, "expand": True, "db_penalty": db_cost, "db_cost": db_cost}
            if not cache_disabled:
                cache["text"].add(text_key)
                if structure_key:
                    cache["structure"].add(structure_key)
                cache["result"].add(result_key)
            cache["stats"]["passed"] += 1
        return {"skip": False, "reason": None, "terminal": False, "expand": True, "db_penalty": db_cost, "db_cost": db_cost}

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
        sql = _extract_sql(response)
        if not sql:
            return []
        if not sql.endswith(";"):
            sql += ";"
        parent_key = _normalize_sql_key(parent_y)
        if parent_key and _normalize_sql_key(sql) == parent_key:
            return []
        return [sql]

    @staticmethod
    def dedup_proposals(proposals: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for p in proposals:
            sql = _extract_sql(p)
            key = _normalize_sql_key(sql)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(sql if sql.endswith(";") else sql + ";")
        return out

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
