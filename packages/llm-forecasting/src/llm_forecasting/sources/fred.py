"""Federal Reserve Economic Data (FRED) source.

FRED provides economic data series that can be used to generate
forecasting questions about whether values will increase or decrease.

Series list matches the original forecastbench FRED questions.
"""

import logging
from datetime import date, datetime, timezone

import httpx

from llm_forecasting.config import settings
from llm_forecasting.models import Question, QuestionType, Resolution, SourceType
from llm_forecasting.sources.base import QuestionSource, registry

logger = logging.getLogger(__name__)

BASE_URL = "https://api.stlouisfed.org/fred"

# FRED series from original forecastbench - curated by domain experts
# See: src/helpers/fred.py in the original forecastbench repo
# This is a subset of the full list for initial implementation
DEFAULT_SERIES = [
    # Financial indicators
    {"id": "DFF", "series_name": "the effective federal funds rate (interest rate)"},
    {"id": "EFFR", "series_name": "the effective federal funds rate (interest rate) set by the Federal Reserve"},
    {"id": "DFEDTARL", "series_name": "the lower limit of the target range of the federal funds rate"},
    {"id": "DFEDTARU", "series_name": "the upper limit of the target range of the federal funds rate"},
    {"id": "DGS10", "series_name": "the market yield on US treasury securities at 10-year constant maturity"},
    {"id": "DGS2", "series_name": "the market yield on US treasury securities at 2-year constant maturity"},
    {"id": "DGS30", "series_name": "the market yield on US treasury securities at 30-year constant maturity"},
    {"id": "T10Y2Y", "series_name": "the yield spread between 10-year and 2-year US Treasury bonds"},
    {"id": "T10Y3M", "series_name": "the yield spread between 10-year and 3-month US Treasury bonds"},
    {"id": "SOFR", "series_name": "the Federal Reserve's Secured Overnight Financing Rate"},
    {"id": "MORTGAGE30US", "series_name": "the 30-year fixed rate mortgage average in the US"},
    {"id": "MORTGAGE15US", "series_name": "the 15-year fixed rate mortgage average in the US"},
    # Market indices
    {"id": "SP500", "series_name": "the S&P 500, which represents the daily index value at market close"},
    {"id": "DJIA", "series_name": "the Dow Jones Industrial Average"},
    {"id": "NASDAQ100", "series_name": "the NASDAQ 100 Index"},
    {"id": "NASDAQCOM", "series_name": "the NASDAQ Composite Index"},
    {"id": "VIXCLS", "series_name": "the Chicago Board Options Exchange's Volatility Index"},
    {"id": "NIKKEI225", "series_name": "the Nikkei 225 Stock Average"},
    # Commodities
    {"id": "DCOILWTICO", "series_name": "the price of West Texas Intermediate (WTI - Cushing) crude oil"},
    {"id": "DCOILBRENTEU", "series_name": "the price of Brent crude oil"},
    {"id": "DHHNGSP", "series_name": "the spot price of Henry Hub natural gas"},
    {"id": "CBBTCUSD", "series_name": "the price of Bitcoin, as measured by Coinbase"},
    # Exchange rates
    {"id": "DEXUSEU", "series_name": "the spot exchange rate of US dollars to euros"},
    {"id": "DEXUSUK", "series_name": "the spot exchange rate of US dollars to UK pound sterling"},
    {"id": "DEXJPUS", "series_name": "the spot exchange rate of Japanese yen to US dollars"},
    {"id": "DEXCAUS", "series_name": "the spot exchange rate of Canadian dollars to US dollars"},
    {"id": "DEXCHUS", "series_name": "the spot exchange rate of Chinese yuan renminbi to US dollars"},
    # Labor market
    {"id": "UNRATE", "series_name": "the unemployment rate for US civilian labor force"},
    {"id": "CIVPART", "series_name": "the labor force participation rate among US civilians"},
    {"id": "ICSA", "series_name": "the weekly number of initial unemployment claims"},
    {"id": "CCSA", "series_name": "the number of insured unemployment claims"},
    {"id": "EMRATIO", "series_name": "the employment-population ratio for US civilians"},
    {"id": "CE16OV", "series_name": "the number of employed US civilians"},
    {"id": "UNEMPLOY", "series_name": "number of unemployed US civilians"},
    {"id": "MANEMP", "series_name": "the number of US employees in manufacturing"},
    # Inflation expectations
    {"id": "T10YIE", "series_name": "the US' 10-year breakeven inflation rate"},
    {"id": "T5YIE", "series_name": "the US' 5-year breakeven inflation rate"},
    {"id": "EXPINF1YR", "series_name": "the Federal Reserve Bank of Cleveland's 1-year expected inflation rate"},
    {"id": "EXPINF10YR", "series_name": "the Federal Reserve Bank of Cleveland's 10-year expected inflation rate"},
    # Money supply
    {"id": "WM1NS", "series_name": "USD money supply as measured by M1"},
    {"id": "WM2NS", "series_name": "USD money supply as measured by M2"},
    {"id": "WALCL", "series_name": "the total dollar amount of assets held by all US Federal Reserve banks"},
    {"id": "CURRCIR", "series_name": "the number of US dollars in circulation"},
    # Banking
    {"id": "TOTBKCR", "series_name": "the total dollar amount of bank credit held by all US commercial banks"},
    {"id": "DPSACBW027SBOG", "series_name": "the amount of money representing deposits in all US commercial banks"},
    {"id": "TLAACBW027SBOG", "series_name": "the total dollar amount of assets held by all US commercial banks"},
    # Financial conditions
    {"id": "NFCI", "series_name": "the Chicago Fed's National Financial Conditions Index"},
    {"id": "STLFSI4", "series_name": "the St. Louis Fed Financial Stress Index"},
    {"id": "WEI", "series_name": "the Weekly Economic Index (Lewis-Mertens-Stock)"},
    # Corporate bonds
    {"id": "DAAA", "series_name": "Moody's Seasoned Aaa Corporate Bond Yield"},
    {"id": "DBAA", "series_name": "Moody's Seasoned Baa Corporate Bond Yield"},
    {"id": "AAA10Y", "series_name": "Moody's Aaa Corporate Bond Yield compared to the 10-year Treasury yield"},
    {"id": "BAA10Y", "series_name": "Moody's Seasoned Baa Corporate Bond Yield compared to the 10-year Treasury yield"},
    # Consumer
    {"id": "GASREGW", "series_name": "the average price of regular gas in the US"},
    {"id": "GASDESW", "series_name": "the average price of diesel in the US"},
]


@registry.register
class FREDSource(QuestionSource):
    """Fetch economic data from FRED and generate forecasting questions.

    Unlike market sources, FRED questions ask whether a value will
    increase or decrease by a certain date.
    """

    name = "fred"

    def __init__(
        self,
        api_key: str | None = None,
        series_list: list[dict] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize FRED source.

        Args:
            api_key: FRED API key. If not provided, uses settings.fred_api_key.
                    Get a key at https://fred.stlouisfed.org/docs/api/api_key.html
            series_list: List of series to track, each with 'id' and 'series_name' keys.
                        Defaults to curated economic indicators.
            http_client: Optional httpx client.
        """
        self._api_key = api_key if api_key is not None else settings.fred_api_key
        self._series_list = series_list or DEFAULT_SERIES
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _get_series_info(self, series_id: str) -> dict | None:
        """Get metadata about a series."""
        if not self._api_key:
            logger.warning("FRED API key not set")
            return None

        client = await self._get_client()
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        response = await client.get(f"{BASE_URL}/series", params=params)
        if not response.is_success:
            return None

        data = response.json()
        seriess = data.get("seriess", [])
        return seriess[0] if seriess else None

    async def _get_observations(
        self,
        series_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get recent observations for a series."""
        if not self._api_key:
            return []

        client = await self._get_client()
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        response = await client.get(f"{BASE_URL}/series/observations", params=params)
        if not response.is_success:
            return []

        data = response.json()
        return data.get("observations", [])

    def _generate_question_text(self, series_name: str) -> str:
        """Generate question text for a series."""
        return (
            f"Will {series_name} have increased by {{resolution_date}} "
            f"as compared to its value on {{forecast_due_date}}?"
        )

    async def _series_to_question(self, series_config: dict) -> Question | None:
        """Convert a FRED series to a Question."""
        series_id = series_config["id"]
        series_name = series_config.get("series_name") or series_config.get("name", series_id)

        # Get series info
        series_info = await self._get_series_info(series_id)
        if not series_info:
            logger.warning(f"Could not fetch series info for {series_id}")
            return None

        # Get latest observation
        observations = await self._get_observations(series_id, limit=1)
        if not observations:
            logger.warning(f"No observations for {series_id}")
            return None

        latest_obs = observations[0]
        latest_value = latest_obs.get("value")
        if latest_value == "." or latest_value is None:
            return None

        try:
            current_value = float(latest_value)
        except ValueError:
            return None

        # Build background from series metadata
        background = (
            f"Series: {series_info.get('title', series_name)}. "
            f"Units: {series_info.get('units', 'N/A')}. "
            f"Frequency: {series_info.get('frequency', 'N/A')}. "
            f"Seasonal adjustment: {series_info.get('seasonal_adjustment', 'N/A')}. "
            f"Notes: {series_info.get('notes', 'N/A')[:500]}"
        )

        return Question(
            id=series_id,
            source=self.name,
            source_type=SourceType.DATA,
            text=self._generate_question_text(series_name),
            background=background,
            url=f"https://fred.stlouisfed.org/series/{series_id}",
            question_type=QuestionType.BINARY,
            created_at=datetime.now(timezone.utc),
            resolution_date=None,  # Set when creating question sets
            resolved=False,
            base_rate=current_value,  # Current value for reference
        )

    async def fetch_questions(self) -> list[Question]:
        """Fetch questions for all configured FRED series."""
        questions = []

        for series_config in self._series_list:
            try:
                q = await self._series_to_question(series_config)
                if q:
                    questions.append(q)
            except httpx.HTTPError as e:
                logger.warning(f"Failed to fetch series {series_config['id']}: {e}")

        logger.info(f"Fetched {len(questions)} questions from FRED")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch the current value for a FRED series.

        For FRED questions, the resolution depends on comparing the
        current value to the value at forecast time. This method
        returns the current value.
        """
        observations = await self._get_observations(question_id, limit=1)
        if not observations:
            return None

        latest_obs = observations[0]
        latest_value = latest_obs.get("value")
        if latest_value == "." or latest_value is None:
            return None

        try:
            value = float(latest_value)
        except ValueError:
            return None

        obs_date = latest_obs.get("date")
        if obs_date:
            try:
                resolution_date = datetime.strptime(obs_date, "%Y-%m-%d").date()
            except ValueError:
                resolution_date = date.today()
        else:
            resolution_date = date.today()

        return Resolution(
            question_id=question_id,
            source=self.name,
            date=resolution_date,
            value=value,
        )

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
