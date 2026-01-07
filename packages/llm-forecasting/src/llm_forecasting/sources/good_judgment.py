"""Good Judgment Open source.

Good Judgment Open is a public forecasting platform with questions on
geopolitics, economics, science, and more.

Note: GJO does not have a public API. This source scrapes the public HTML pages.
"""

import asyncio
import logging
import re
from datetime import date, datetime, timezone

import httpx
from bs4 import BeautifulSoup

from llm_forecasting.models import Question, QuestionType, Resolution, SourceType
from llm_forecasting.sources.base import QuestionSource, registry

logger = logging.getLogger(__name__)

BASE_URL = "https://www.gjopen.com"

# Rate limiting: delay between requests (seconds)
REQUEST_DELAY = 1.0  # GJO is strict about rate limiting
MAX_RETRIES = 3
RETRY_DELAY = 5.0  # Wait longer on rate limit


@registry.register
class GoodJudgmentSource(QuestionSource):
    """Fetch questions from Good Judgment Open via HTML scraping.

    Good Judgment Open is a public forecasting platform run by Good Judgment Inc.,
    founded by Philip Tetlock (author of Superforecasting).

    Since GJO doesn't have a public API, we scrape the HTML pages.
    """

    name = "good_judgment"

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "ForecastBench/2.0 (forecasting research)"},
            )
        return self._client

    async def _fetch_with_retry(self, url: str, params: dict | None = None) -> httpx.Response | None:
        """Fetch URL with retry logic for rate limiting."""
        client = await self._get_client()

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.get(url, params=params)

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = RETRY_DELAY * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code == 200:
                    return response

                logger.warning(f"HTTP {response.status_code} for {url}")
                return None

            except httpx.HTTPError as e:
                logger.warning(f"HTTP error for {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return None

        return None

    async def _fetch_question_ids_from_page(self, page: int = 1) -> list[str]:
        """Fetch question IDs from a questions list page."""
        response = await self._fetch_with_retry(
            f"{BASE_URL}/questions",
            params={"status": "active", "page": page},
        )
        if not response:
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        # Find question links - format: /questions/4943-slug or full URL
        pattern = re.compile(r"(?:https://www\.gjopen\.com)?/questions/(\d+)")
        question_ids = []

        for link in soup.find_all("a", href=pattern):
            href = link.get("href", "")
            match = pattern.search(href)
            if match:
                qid = match.group(1)
                if qid not in question_ids:
                    question_ids.append(qid)

        return question_ids

    async def _fetch_question_page(self, question_id: str) -> dict | None:
        """Fetch and parse a single question's HTML page."""
        response = await self._fetch_with_retry(f"{BASE_URL}/questions/{question_id}")
        if not response:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text().lower()

        # Extract title from h3 (GJO's format)
        title = None
        for tag in ["h3", "h1", "h2"]:
            heading = soup.find(tag)
            if heading:
                text = heading.get_text(strip=True)
                if text and len(text) > 20 and "sign up" not in text.lower():
                    title = text
                    break

        if not title:
            logger.warning(f"No title found for question {question_id}")
            return None

        # Extract description/background
        background = None
        for div in soup.find_all("div", class_=lambda x: x and "background" in str(x).lower()):
            text = div.get_text(strip=True)
            if text and len(text) > 50:
                background = text
                break

        # Extract probabilities (percentages in the page)
        probabilities = []
        for text in soup.stripped_strings:
            pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
            if pct_match:
                pct = float(pct_match.group(1))
                if 0 < pct < 100 and pct not in probabilities:
                    probabilities.append(pct)

        # For binary questions, take the first probability as Yes probability
        # For multi-choice, we'll take the highest as the "base rate"
        base_rate = None
        if probabilities:
            # Convert from percentage to decimal
            base_rate = max(probabilities) / 100.0

        # Extract close date
        close_date = None
        date_pattern = re.compile(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}",
            re.IGNORECASE,
        )
        for text in soup.stripped_strings:
            match = date_pattern.search(text)
            if match:
                try:
                    # Parse the date string
                    date_str = match.group(0).replace(",", "")
                    close_date = datetime.strptime(date_str, "%b %d %Y").date()
                except ValueError:
                    try:
                        close_date = datetime.strptime(date_str, "%B %d %Y").date()
                    except ValueError:
                        pass
                if close_date:
                    break

        # Check resolution status
        resolved = "resolved" in page_text or "this question is closed" in page_text
        resolution_value = None

        if resolved:
            # Try to find resolution outcome
            if "resolved: yes" in page_text or "outcome: yes" in page_text:
                resolution_value = 1.0
            elif "resolved: no" in page_text or "outcome: no" in page_text:
                resolution_value = 0.0

        # Determine question type
        # Multi-choice questions typically have multiple options listed
        question_type = QuestionType.BINARY
        if len(probabilities) > 2:
            # Likely multi-choice, but we'll still treat as binary for now
            # (using highest probability as base rate)
            pass

        return {
            "id": question_id,
            "title": title,
            "background": background,
            "base_rate": base_rate,
            "close_date": close_date,
            "resolved": resolved,
            "resolution_value": resolution_value,
            "question_type": question_type,
            "probabilities": probabilities,
        }

    async def fetch_questions(self, max_questions: int | None = None) -> list[Question]:
        """Fetch open questions from Good Judgment Open.

        Args:
            max_questions: Maximum number of questions to fetch (None for all).
        """
        questions = []
        page = 1
        max_pages = 10  # Limit to avoid scraping too much

        while page <= max_pages:
            logger.debug(f"Fetching question IDs from page {page}")
            question_ids = await self._fetch_question_ids_from_page(page)

            if not question_ids:
                break

            # Fetch each question's details
            for qid in question_ids:
                if max_questions and len(questions) >= max_questions:
                    break

                await asyncio.sleep(REQUEST_DELAY)  # Rate limiting

                data = await self._fetch_question_page(qid)
                if data:
                    try:
                        question = Question(
                            id=data["id"],
                            source=self.name,
                            source_type=SourceType.MARKET,
                            text=data["title"],
                            background=data["background"],
                            url=f"{BASE_URL}/questions/{data['id']}",
                            question_type=data["question_type"],
                            created_at=datetime.now(timezone.utc),
                            resolution_date=data["close_date"],
                            resolved=data["resolved"],
                            resolution_value=data["resolution_value"],
                            base_rate=data["base_rate"],
                        )
                        questions.append(question)
                    except Exception as e:
                        logger.warning(f"Failed to create Question for {qid}: {e}")

            if max_questions and len(questions) >= max_questions:
                break

            page += 1
            await asyncio.sleep(REQUEST_DELAY)

        logger.info(f"Fetched {len(questions)} questions from Good Judgment Open")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution for a specific question.

        Returns the current crowd forecast or resolved value.
        """
        data = await self._fetch_question_page(question_id)
        if not data:
            return None

        # If resolved, return the resolution value
        if data["resolved"] and data["resolution_value"] is not None:
            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=data["resolution_value"],
            )

        # Otherwise return current crowd forecast
        if data["base_rate"] is not None:
            return Resolution(
                question_id=question_id,
                source=self.name,
                date=date.today(),
                value=data["base_rate"],
            )

        return None

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
