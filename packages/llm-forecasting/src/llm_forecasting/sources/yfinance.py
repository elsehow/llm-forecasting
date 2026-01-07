"""Yahoo Finance data source.

Provides stock price and financial market data for generating
forecasting questions about price movements.

Uses the yfinance library which handles Yahoo Finance API authentication.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import yfinance as yf

from llm_forecasting.models import Question, QuestionType, Resolution, SourceType
from llm_forecasting.sources.base import QuestionSource, registry

logger = logging.getLogger(__name__)

# Default tickers to track
DEFAULT_TICKERS = [
    {"symbol": "^GSPC", "name": "S&P 500"},
    {"symbol": "^DJI", "name": "Dow Jones Industrial Average"},
    {"symbol": "^IXIC", "name": "NASDAQ Composite"},
    {"symbol": "^VIX", "name": "CBOE Volatility Index"},
    {"symbol": "GC=F", "name": "Gold Futures"},
    {"symbol": "CL=F", "name": "Crude Oil Futures"},
    {"symbol": "BTC-USD", "name": "Bitcoin USD"},
    {"symbol": "ETH-USD", "name": "Ethereum USD"},
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
]


@registry.register
class YahooFinanceSource(QuestionSource):
    """Fetch financial data from Yahoo Finance.

    Questions ask whether a stock/index price will increase or decrease
    by a certain date compared to its current value.
    """

    name = "yfinance"

    def __init__(
        self,
        tickers: list[dict] | None = None,
    ):
        """Initialize Yahoo Finance source.

        Args:
            tickers: List of tickers to track, each with 'symbol' and 'name' keys.
        """
        self._tickers = tickers or DEFAULT_TICKERS
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _get_quote_sync(self, symbol: str) -> dict | None:
        """Get current quote for a symbol (synchronous)."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info or info.get("regularMarketPrice") is None:
                # Try fast_info as fallback
                fast = ticker.fast_info
                if fast:
                    return {
                        "regularMarketPrice": fast.get("lastPrice"),
                        "regularMarketPreviousClose": fast.get("previousClose"),
                        "regularMarketDayHigh": fast.get("dayHigh"),
                        "regularMarketDayLow": fast.get("dayLow"),
                        "marketCap": fast.get("marketCap"),
                        "shortName": symbol,
                    }
                return None
            return info
        except Exception as e:
            logger.warning(f"Failed to fetch quote for {symbol}: {e}")
            return None

    async def _get_quote(self, symbol: str) -> dict | None:
        """Get current quote for a symbol (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._get_quote_sync, symbol)

    def _generate_question_text(self, ticker_name: str) -> str:
        """Generate question text for a ticker."""
        return (
            f"Will {ticker_name} have increased by {{resolution_date}} "
            f"as compared to its value on {{forecast_due_date}}?"
        )

    def _quote_to_question(self, ticker_config: dict, quote: dict) -> Question | None:
        """Convert a Yahoo Finance quote to a Question."""
        symbol = ticker_config["symbol"]
        name = ticker_config["name"]

        # Get current price
        price = quote.get("regularMarketPrice") or quote.get("currentPrice")
        if price is None:
            return None

        # Build background
        market_cap = quote.get("marketCap")
        prev_close = quote.get("regularMarketPreviousClose")
        day_high = quote.get("regularMarketDayHigh")
        day_low = quote.get("regularMarketDayLow")
        volume = quote.get("regularMarketVolume")

        background_parts = [
            f"Symbol: {symbol}",
            f"Name: {quote.get('longName') or quote.get('shortName') or name}",
            f"Current price: {price}",
        ]
        if prev_close:
            background_parts.append(f"Previous close: {prev_close}")
        if day_high and day_low:
            background_parts.append(f"Day range: {day_low} - {day_high}")
        if market_cap:
            background_parts.append(f"Market cap: {market_cap:,}")
        if volume:
            background_parts.append(f"Volume: {volume:,}")

        background = ". ".join(background_parts)

        return Question(
            id=symbol,
            source=self.name,
            source_type=SourceType.DATA,
            text=self._generate_question_text(name),
            background=background,
            url=f"https://finance.yahoo.com/quote/{symbol}",
            question_type=QuestionType.BINARY,
            created_at=datetime.now(timezone.utc),
            resolution_date=None,  # Set when creating question sets
            resolved=False,
            base_rate=price,  # Current price for reference
        )

    async def fetch_questions(self) -> list[Question]:
        """Fetch questions for all configured tickers."""
        questions = []

        for ticker_config in self._tickers:
            symbol = ticker_config["symbol"]
            quote = await self._get_quote(symbol)
            if quote:
                q = self._quote_to_question(ticker_config, quote)
                if q:
                    questions.append(q)

        logger.info(f"Fetched {len(questions)} questions from Yahoo Finance")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch the current price for a ticker.

        For Yahoo Finance questions, the resolution depends on comparing
        the current price to the price at forecast time.
        """
        quote = await self._get_quote(question_id)
        if not quote:
            return None

        price = quote.get("regularMarketPrice") or quote.get("currentPrice")
        if price is None:
            return None

        return Resolution(
            question_id=question_id,
            source=self.name,
            date=date.today(),
            value=float(price),
        )

    async def close(self):
        self._executor.shutdown(wait=False)
