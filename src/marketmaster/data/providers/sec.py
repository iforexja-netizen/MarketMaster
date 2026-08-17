"""
SEC EDGAR Data Provider

Fetches filing metadata and structured fundamental data (XBRL) from SEC EDGAR.

Endpoints:
- EFTS: efts.sec.gov/LATEST/search-index?q=... (filing search)
- Data API: data.sec.gov/api/xbrlcompanyconcept/... (structured XBRL data)
- Submissions: data.sec.gov/submissions/CIK{padded}.json (filing history)

SEC requires a User-Agent header identifying your app and email.
"""

from datetime import date
from typing import Any, Optional

import httpx

from marketmaster.data.providers.base import DataProvider


class SecEdgarProvider(DataProvider):
    """SEC EDGAR data provider for filings and structured fundamentals."""

    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    DATA_URL = "https://data.sec.gov"

    def __init__(self, user_agent: str):
        """user_agent must be 'CompanyName email@example.com' per SEC policy."""
        self.user_agent = user_agent

    @property
    def name(self) -> str:
        return "sec_edgar"

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

    async def health_check(self) -> bool:
        """Check if SEC EDGAR is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.DATA_URL}/submissions/CIK0000320193.json",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def fetch_filings(
        self,
        cik: str,
        form_types: list[str],
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        """
        Fetch SEC filing metadata for a CIK.

        Returns normalized dicts with:
        cik, accession_no, filing_date, form_type, description, primary_document, filing_url
        """
        # Pad CIK to 10 digits
        padded_cik = cik.zfill(10)
        filings: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.DATA_URL}/submissions/CIK{padded_cik}.json",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNo", [])
            docs = recent.get("primaryDocument", [])
            descriptions = recent.get("primaryDocDescription", [])

            for i in range(len(forms)):
                if forms[i] not in form_types:
                    continue

                filing_date_str = dates[i] if i < len(dates) else ""
                if not filing_date_str:
                    continue

                filing_date = date.fromisoformat(filing_date_str[:10])
                if filing_date < start or filing_date > end:
                    continue

                accession = accessions[i] if i < len(accessions) else ""
                accession_no = accession.replace("-", "")
                doc = docs[i] if i < len(docs) else ""

                filings.append({
                    "cik": cik,
                    "accession_no": accession,
                    "filing_date": filing_date,
                    "form_type": forms[i],
                    "description": descriptions[i] if i < len(descriptions) else None,
                    "primary_document": doc,
                    "filing_url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no}/{doc}",
                })

        return filings

    async def fetch_fundamentals(
        self,
        cik: str,
        concept: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch structured fundamental data from SEC XBRL API.

        Uses data.sec.gov/api/xbrlcompanyconcept to get XBRL-tagged financial data.

        If concept is None, fetches a default set of key concepts:
        Revenues, NetIncomeLoss, Assets, Liabilities, StockholdersEquity,
        CashAndCashEquivalents, OperatingIncomeLoss, EarningsPerShareBasic

        Returns normalized dicts with:
        security_id (to be resolved), report_date, statement_type, items
        """
        padded_cik = cik.zfill(10)

        default_concepts = [
            ("us-gaap", "Revenues"),
            ("us-gaap", "NetIncomeLoss"),
            ("us-gaap", "Assets"),
            ("us-gaap", "Liabilities"),
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
            ("us-gaap", "OperatingIncomeLoss"),
            ("us-gaap", "EarningsPerShareBasic"),
            ("us-gaap", "LongTermDebt"),
            ("us-gaap", "InventoryNet"),
            ("us-gaap", "AccountsReceivableNetCurrent"),
            ("us-gaap", "CommonStockSharesOutstanding"),
        ]

        concepts_to_fetch = (
            [("us-gaap", concept)] if concept else default_concepts
        )

        all_data: dict[str, Any] = {}  # {report_date: {concept: value}}

        async with httpx.AsyncClient(timeout=30) as client:
            for taxonomy, concept_name in concepts_to_fetch:
                url = (
                    f"{self.DATA_URL}/api/xbrlcompanyconcept/CIK{padded_cik}/"
                    f"{taxonomy}/{concept_name}.json"
                )
                try:
                    resp = await client.get(url, headers=self._headers())
                    if resp.status_code != 200:
                        continue
                    data = resp.json()

                    units = data.get("units", {})
                    # USD for monetary values, shares for share counts
                    unit_data = units.get("USD", units.get("shares", units.get("USD/shares", [])))

                    for item in unit_data if isinstance(unit_data, list) else []:
                        end_date_str = item.get("end")
                        if not end_date_str:
                            continue

                        report_date = end_date_str[:10]
                        if report_date not in all_data:
                            all_data[report_date] = {"report_date": report_date}

                        val = item.get("val")
                        if val is not None:
                            all_data[report_date][concept_name] = float(val)

                except Exception:
                    continue

        # Convert to list of fundamentals
        results: list[dict[str, Any]] = []
        for report_date_str, items in all_data.items():
            results.append({
                "report_date": date.fromisoformat(report_date_str),
                "period_type": "annual",  # SEC XBRL includes both; can be refined
                "statement_type": "xbrl",
                "items": items,
                "source": "sec_edgar",
                "filing_date": None,  # to be filled from filings data
            })

        return results
