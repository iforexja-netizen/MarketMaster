"""
Data Quality Service — Freshness, completeness, and anomaly detection.

Runs validation checks against the Data Plane and logs results to
data_quality_log. Every check is traceable and can trigger alerts.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select, func, text
from sqlalchemy.orm import Session

from marketmaster.db.models import DataQualityLog


@dataclass
class DataQualityResult:
    """Result of a single data quality check."""
    check_name: str
    table_name: str
    passed: bool
    severity: str = "info"  # info, warning, error, critical
    details: dict[str, Any] = field(default_factory=dict)
    check_scope: Optional[str] = None


class DataQualityService:
    """
    Runs data quality checks against the Data Plane.

    Checks:
    - freshness: is the latest data within an expected time window?
    - completeness: are expected fields present and non-null?
    - null_rate: what percentage of a column is null?
    - duplicate_rate: are there unexpected duplicate rows?
    """

    def __init__(self, db: Session):
        self.db = db

    def _log_result(self, result: DataQualityResult) -> None:
        """Log a quality check result to the database."""
        entry = DataQualityLog(
            table_name=result.table_name,
            check_name=result.check_name,
            check_scope=result.check_scope,
            passed=result.passed,
            details=result.details,
            severity=result.severity,
        )
        self.db.add(entry)
        self.db.commit()

    # ── Freshness ────────────────────────────────────────────────────────────

    def check_freshness(
        self,
        table_name: str,
        date_column: str = "date",
        security_id: Optional[int] = None,
        max_age_hours: int = 24,
    ) -> DataQualityResult:
        """
        Check if the latest data in a table is within the expected freshness window.

        Args:
            table_name: Table to check
            date_column: Column containing the date/timestamp
            security_id: Optional filter to a specific security
            max_age_hours: Maximum acceptable age in hours
        """
        scope = f"security_id={security_id}" if security_id else "all"

        query = f"SELECT MAX({date_column}) as latest FROM {table_name}"
        if security_id:
            query += f" WHERE security_id = {security_id}"

        result = self.db.execute(text(query)).first()

        if result is None or result.latest is None:
            res = DataQualityResult(
                check_name="freshness",
                table_name=table_name,
                passed=False,
                severity="error",
                details={"error": "No data found", "max_age_hours": max_age_hours},
                check_scope=scope,
            )
            self._log_result(res)
            return res

        latest = result.latest
        if isinstance(latest, str):
            latest = datetime.fromisoformat(latest)
        elif hasattr(latest, "year") and not isinstance(latest, datetime):
            # Convert date to datetime
            latest = datetime.combine(latest, datetime.min.time())

        # Make timezone-aware
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age = now - latest
        age_hours = age.total_seconds() / 3600

        passed = age_hours <= max_age_hours
        severity = "ok" if passed else ("warning" if age_hours <= max_age_hours * 2 else "error")

        res = DataQualityResult(
            check_name="freshness",
            table_name=table_name,
            passed=passed,
            severity=severity,
            details={
                "latest_date": str(latest),
                "age_hours": round(age_hours, 2),
                "max_age_hours": max_age_hours,
            },
            check_scope=scope,
        )
        self._log_result(res)
        return res

    # ── Completeness ─────────────────────────────────────────────────────────

    def check_completeness(
        self,
        table_name: str,
        required_columns: list[str],
        security_id: Optional[int] = None,
    ) -> DataQualityResult:
        """
        Check if required columns have non-null values.

        Args:
            table_name: Table to check
            required_columns: Columns that must not be null
            security_id: Optional filter to a specific security
        """
        scope = f"security_id={security_id}" if security_id else "all"
        where_clause = f"WHERE security_id = {security_id}" if security_id else ""
        total_query = f"SELECT COUNT(*) FROM {table_name} {where_clause}"
        total = self.db.execute(text(total_query)).scalar() or 0

        null_counts: dict[str, int] = {}
        for col in required_columns:
            null_query = f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL"
            if security_id:
                null_query += f" AND security_id = {security_id}"
            null_count = self.db.execute(text(null_query)).scalar() or 0
            null_counts[col] = null_count

        total_nulls = sum(null_counts.values())
        passed = total_nulls == 0 and total > 0

        res = DataQualityResult(
            check_name="completeness",
            table_name=table_name,
            passed=passed,
            severity="warning" if total_nulls > 0 else "info",
            details={
                "total_rows": total,
                "null_counts": null_counts,
                "required_columns": required_columns,
            },
            check_scope=scope,
        )
        self._log_result(res)
        return res

    # ── Null Rate ────────────────────────────────────────────────────────────

    def check_null_rate(
        self,
        table_name: str,
        column_name: str,
        max_null_rate: float = 0.05,
    ) -> DataQualityResult:
        """
        Check the null rate of a specific column.

        Args:
            table_name: Table to check
            column_name: Column to check
            max_null_rate: Maximum acceptable null rate (0.0-1.0)
        """
        total = self.db.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0
        nulls = self.db.execute(
            text(f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL")
        ).scalar() or 0

        null_rate = nulls / total if total > 0 else 1.0
        passed = null_rate <= max_null_rate

        res = DataQualityResult(
            check_name="null_rate",
            table_name=table_name,
            passed=passed,
            severity="error" if null_rate > 0.1 else "warning",
            details={
                "column": column_name,
                "null_count": nulls,
                "total_count": total,
                "null_rate": round(null_rate, 4),
                "max_null_rate": max_null_rate,
            },
            check_scope=column_name,
        )
        self._log_result(res)
        return res

    # ── Duplicate Rate ───────────────────────────────────────────────────────

    def check_duplicates(
        self,
        table_name: str,
        key_columns: list[str],
    ) -> DataQualityResult:
        """
        Check for duplicate rows based on key columns.

        Args:
            table_name: Table to check
            key_columns: Columns that should form a unique key
        """
        key_str = ", ".join(key_columns)
        query = text(f"""
            SELECT {key_str}, COUNT(*) as cnt
            FROM {table_name}
            GROUP BY {key_str}
            HAVING COUNT(*) > 1
            LIMIT 100
        """)

        duplicates = self.db.execute(query).fetchall()

        total_dups = len(duplicates)
        passed = total_dups == 0

        res = DataQualityResult(
            check_name="duplicate_check",
            table_name=table_name,
            passed=passed,
            severity="error" if total_dups > 0 else "info",
            details={
                "key_columns": key_columns,
                "duplicate_groups": total_dups,
                "sample_duplicates": [
                    {col: getattr(row, col, None) for col in key_columns}
                    for row in duplicates[:10]
                ],
            },
            check_scope=key_str,
        )
        self._log_result(res)
        return res

    # ── Run All Checks ───────────────────────────────────────────────────────

    def run_all_checks(self) -> list[DataQualityResult]:
        """Run the standard battery of data quality checks."""
        results: list[DataQualityResult] = []

        # OHLCV daily freshness
        results.append(
            self.check_freshness("ohlcv_daily", "date", max_age_hours=24)
        )

        # OHLCV daily completeness
        results.append(
            self.check_completeness(
                "ohlcv_daily",
                ["open", "high", "low", "close", "volume"],
            )
        )

        # Macro series freshness
        results.append(
            self.check_freshness("macro_series", "observation_date", max_age_hours=168)
        )

        # MCEI freshness
        results.append(
            self.check_freshness("mcei_history", "as_of_date", max_age_hours=24)
        )

        # OHLCV daily duplicates
        results.append(
            self.check_duplicates("ohlcv_daily", ["security_id", "date"])
        )

        # Macro series duplicates
        results.append(
            self.check_duplicates("macro_series", ["series_code", "observation_date"])
        )

        return results
