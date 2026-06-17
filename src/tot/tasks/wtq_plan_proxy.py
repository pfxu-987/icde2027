import math
import re
import sqlite3
import threading
from typing import Any

from tot.tasks.bird import _extract_sql, _is_safe_select, _normalize_sql_key, _rows_hash, _sql_clause_signature
from tot.tasks.wtq import WTQDecompRepairTask


def _p95(values: list[float]) -> float:
    if not values:
        return 1.0
    xs = sorted(float(v) for v in values)
    idx = int(math.ceil(0.95 * len(xs))) - 1
    return max(1e-6, xs[max(0, min(idx, len(xs) - 1))])


class WTQPlanProxyTask(WTQDecompRepairTask):
    """WTQ decomp-repair task aligned with Bird plan-proxy scheduling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proxy_lock = threading.Lock()
        self._table_count_cache: dict[tuple[str, str], int] = {}
        self._proxy_costs_by_db: dict[str, list[float]] = {}
        self._proxy_bmax_by_db: dict[str, float] = {}
        self._proxy_global_bmax = 1.0
        self._proxy_stats = {
            "calls": 0,
            "explain_failures": 0,
            "hard_pruned": 0,
            "cost_sum": 0.0,
            "normalized_cost_sum": 0.0,
            "full_scans": 0,
            "temp_btrees": 0,
            "joins": 0,
        }
        self._proxy_calibration = {
            "mode": None,
            "n_databases": 0,
            "n_budget_samples": 0,
        }

    def calibrate_proxy_from_schema(self, db_ids: list[str] | None = None) -> dict[str, Any]:
        if db_ids is None:
            db_ids = sorted({self._db_id(i) for i in range(len(self.data))})

        costs_by_db: dict[str, list[float]] = {}
        for db_id in db_ids:
            table_counts = self._db_table_counts(db_id)
            if not table_counts:
                continue
            costs = self._schema_proxy_budget_samples(table_counts)
            if costs:
                costs_by_db[str(db_id)] = costs

        all_costs: list[float] = []
        with self._proxy_lock:
            for db_id, values in costs_by_db.items():
                self._proxy_bmax_by_db[str(db_id)] = _p95(values)
                all_costs.extend(values)
            if all_costs:
                self._proxy_global_bmax = _p95(all_costs)
            self._proxy_calibration = {
                "mode": "schema_all",
                "n_databases": len(costs_by_db),
                "n_budget_samples": len(all_costs),
            }
        return self.proxy_cost_stats()

    def _db_table_names(self, db_id: str) -> list[str]:
        try:
            with sqlite3.connect(self._db_path(db_id), timeout=10.0) as conn:
                conn.execute("PRAGMA query_only = ON")
                cur = conn.cursor()
                cur.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
                return [str(r[0]) for r in cur.fetchall()]
        except Exception:
            return []

    def _db_table_counts(self, db_id: str) -> list[int]:
        counts: list[int] = []
        for table in self._db_table_names(db_id):
            counts.append(max(0, int(self._table_row_count(db_id, table))))
        return counts

    def _schema_proxy_budget_samples(self, table_counts: list[int]) -> list[float]:
        counts = sorted([max(0, int(c)) for c in table_counts], reverse=True)
        costs: list[float] = []
        for rows in counts:
            base = math.log1p(rows)
            costs.append(base + 1.0)
            costs.append(base + 2.0)

        top = counts[: min(8, len(counts))]
        for i, left in enumerate(top):
            for right in top[i + 1 :]:
                base = math.log1p(left + right)
                costs.append(base + 3.0)
                costs.append(base + 4.0)

        return costs or [1.0]

    def _table_row_count(self, db_id: str, table: str) -> int:
        key = (str(db_id), str(table).lower())
        with self._proxy_lock:
            cached = self._table_count_cache.get(key)
            if cached is not None:
                return int(cached)

        count = 0
        try:
            with sqlite3.connect(self._db_path(db_id), timeout=10.0) as conn:
                conn.execute("PRAGMA query_only = ON")
                cur = conn.cursor()
                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                row = cur.fetchone()
                count = int(row[0] or 0) if row else 0
        except Exception:
            count = 0

        with self._proxy_lock:
            self._table_count_cache[key] = int(count)
        return int(count)

    def _proxy_cost(self, db_id: str, sql: str) -> tuple[bool, float, dict[str, Any]]:
        query = _extract_sql(sql)
        if not _is_safe_select(query):
            return False, 0.0, {"reason": "unsafe_or_empty_sql"}

        try:
            with sqlite3.connect(self._db_path(db_id), timeout=10.0) as conn:
                conn.execute("PRAGMA query_only = ON")
                cur = conn.cursor()
                cur.execute("EXPLAIN QUERY PLAN " + query)
                rows = cur.fetchall()
        except Exception as exc:
            return False, 0.0, {"reason": "explain_failure", "error": str(exc)}

        full_scan_tables: set[str] = set()
        n_scan = 0
        n_temp = 0
        n_access = 0
        details: list[str] = []
        for row in rows:
            detail = str(row[3] if len(row) > 3 else "")
            details.append(detail)
            upper = detail.upper()
            if "USE TEMP B-TREE" in upper:
                n_temp += 1
            if upper.startswith("SCAN "):
                n_scan += 1
                n_access += 1
                match = re.search(r"\bSCAN\s+(?:TABLE\s+)?[`\"[]?([A-Za-z_][A-Za-z0-9_]*)", detail, flags=re.IGNORECASE)
                if match:
                    full_scan_tables.add(match.group(1))
            elif upper.startswith("SEARCH "):
                n_access += 1

        n_join = max(0, n_access - 1)
        rows_scanned = sum(self._table_row_count(db_id, table) for table in full_scan_tables)
        cost = math.log1p(max(0, rows_scanned)) + float(n_scan) + float(n_temp) + float(n_join)
        meta = {
            "details": details,
            "rows_scanned": int(rows_scanned),
            "n_scan": int(n_scan),
            "n_temp": int(n_temp),
            "n_join": int(n_join),
        }
        return True, float(cost), meta

    def _proxy_bmax(self, db_id: str, observed_cost: float) -> float:
        with self._proxy_lock:
            bmax = self._proxy_bmax_by_db.get(str(db_id)) or self._proxy_global_bmax
            if not bmax or bmax <= 1e-6:
                values = self._proxy_costs_by_db.get(str(db_id), [])
                bmax = _p95(values) if values else max(1.0, float(observed_cost))
        return max(1e-6, float(bmax))

    def _record_proxy_cost(self, db_id: str, cost: float, normalized: float, meta: dict[str, Any], hard_pruned: bool = False) -> None:
        with self._proxy_lock:
            self._proxy_stats["calls"] += 1
            if hard_pruned:
                self._proxy_stats["hard_pruned"] += 1
            self._proxy_stats["cost_sum"] += float(cost)
            self._proxy_stats["normalized_cost_sum"] += float(normalized)
            self._proxy_stats["full_scans"] += int(meta.get("n_scan") or 0)
            self._proxy_stats["temp_btrees"] += int(meta.get("n_temp") or 0)
            self._proxy_stats["joins"] += int(meta.get("n_join") or 0)
            self._proxy_costs_by_db.setdefault(str(db_id), []).append(float(cost))

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
                    "db_penalty": 1.0,
                    "db_cost": 1.0,
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
                    "db_penalty": 0.0,
                    "db_cost": 0.0,
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
                    "db_penalty": 0.0,
                    "db_cost": 0.0,
                }

        if not db_id:
            return None

        proxy_ok, proxy_cost, proxy_meta = self._proxy_cost(db_id, sql)
        if not proxy_ok:
            with self._proxy_lock:
                self._proxy_stats["explain_failures"] += 1
            self._record_proxy_cost(db_id, 1.0, 1.0, proxy_meta, hard_pruned=True)
            with self._semantic_cache_lock:
                if not cache_disabled:
                    cache["text"].add(text_key)
                cache["stats"]["invalid_sql"] += 1
                cache["stats"]["terminal"] += 1
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

        ok, rows, _error, _exec_meta = self._execute_sql_with_meta(db_id, sql)
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
                "db_penalty": proxy_norm,
                "db_cost": proxy_norm,
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
                    "db_penalty": proxy_norm,
                    "db_cost": proxy_norm,
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
            "db_penalty": proxy_norm,
            "db_cost": proxy_norm,
        }

    def proxy_cost_stats(self) -> dict[str, Any]:
        with self._proxy_lock:
            stats = dict(self._proxy_stats)
            calls = int(stats.get("calls") or 0)
            stats["avg_cost"] = float(stats.get("cost_sum") or 0.0) / calls if calls else 0.0
            stats["avg_normalized_cost"] = float(stats.get("normalized_cost_sum") or 0.0) / calls if calls else 0.0
            stats["n_calibrated_dbs"] = len(self._proxy_bmax_by_db)
            stats["global_bmax"] = float(self._proxy_global_bmax)
            stats["calibration"] = dict(self._proxy_calibration)
        return stats
