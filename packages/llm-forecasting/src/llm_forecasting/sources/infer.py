"""INFER (RAND Forecasting Initiative) question source.

INFER is a prediction market platform run by the RAND Corporation for
geopolitical and national security forecasting.
"""

import logging
import os
from datetime import date, datetime, timezone

import httpx

from llm_forecasting.date_utils import parse_iso_datetime
from llm_forecasting.http_utils import HTTPClientMixin
from llm_forecasting.models import Question, QuestionType, Resolution, SourceType
from llm_forecasting.sources.base import QuestionSource, registry

logger = logging.getLogger(__name__)

BASE_URL = "https://www.randforecastinginitiative.org"


@registry.register
class INFERSource(QuestionSource, HTTPClientMixin):
    """Fetch questions from INFER prediction market."""

    name = "infer"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize INFER source.

        Args:
            api_key: INFER API key. If not provided, uses INFER_API_KEY env var.
            http_client: Optional HTTP client for testing.
        """
        self._api_key = api_key or os.environ.get("INFER_API_KEY")
        self._client = http_client

    def _get_headers(self) -> dict:
        """Get authorization headers."""
        if not self._api_key:
            raise ValueError("INFER API key required. Set INFER_API_KEY env var.")
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _fetch_page(
        self,
        page: int = 0,
        status: str = "active",
        ids: list[str] | None = None,
    ) -> list[dict]:
        """Fetch a page of questions from INFER API."""
        client = await self._get_client()
        params = {"page": page, "status": status}
        if ids:
            params["ids"] = ",".join(ids)

        response = await client.get(
            f"{BASE_URL}/api/v1/questions",
            params=params,
            headers=self._get_headers(),
        )
        response.raise_for_status()
        return response.json().get("questions", [])

    async def _fetch_all_questions(
        self,
        status: str = "active",
        ids: list[str] | None = None,
    ) -> list[dict]:
        """Fetch all questions, paginating through results."""
        all_questions = []
        seen_ids = set()
        page = 0

        while True:
            questions = await self._fetch_page(page=page, status=status, ids=ids)
            if not questions:
                break

            for q in questions:
                if q["id"] not in seen_ids:
                    seen_ids.add(q["id"])
                    all_questions.append(q)

            page += 1

        return all_questions

    def _question_to_model(self, q: dict) -> Question | None:
        """Convert INFER question dict to Question model."""
        # Only include binary (Yes/No) questions
        if q.get("type") != "Forecast::YesNoQuestion":
            return None

        # Must have at least one prediction
        answers = q.get("answers", [])
        if len(answers) != 2:
            return None

        predictions_count = sum(a.get("predictions_count", 0) for a in answers)
        if predictions_count == 0:
            return None

        # Extract probability (Yes answer)
        probability = None
        for answer in answers:
            if answer.get("name", "").lower() == "yes":
                probability = answer.get("probability")
                break

        # Parse dates
        created_at = parse_iso_datetime(q.get("scoring_start_time"))
        if not created_at:
            created_at = datetime.now(timezone.utc)

        # Resolution date is the earlier of scoring_end_time or ends_at
        scoring_end = parse_iso_datetime(q.get("scoring_end_time"))
        ends_at = parse_iso_datetime(q.get("ends_at"))
        resolution_date = None
        if scoring_end and ends_at:
            resolution_date = min(scoring_end, ends_at).date()
        elif scoring_end:
            resolution_date = scoring_end.date()
        elif ends_at:
            resolution_date = ends_at.date()

        # Check if resolved
        resolved = q.get("resolved?", False)
        resolution_value = None
        if resolved:
            # Get the resolved answer
            for answer in answers:
                if answer.get("name", "").lower() == "yes":
                    # If Yes answer is 100%, resolution is YES
                    if answer.get("probability") == 1.0:
                        resolution_value = 1.0
                    elif answer.get("probability") == 0.0:
                        resolution_value = 0.0
                    # Otherwise ambiguous
                    break

        # Build description from clarifications
        background = q.get("description", "")
        clarifications = q.get("clarifications", [])
        if clarifications:
            criteria = " ".join(c.get("content", "") for c in clarifications)
            if criteria:
                background = f"{background}\n\nResolution criteria: {criteria}"

        return Question(
            id=str(q["id"]),
            source=self.name,
            source_type=SourceType.MARKET,
            text=q.get("name", ""),
            background=background.strip() or None,
            url=f"{BASE_URL}/questions/{q['id']}",
            question_type=QuestionType.BINARY,
            created_at=created_at,
            resolution_date=resolution_date,
            resolved=resolved,
            resolution_value=resolution_value,
            base_rate=probability,
        )

    async def fetch_questions(self) -> list[Question]:
        """Fetch active binary questions from INFER."""
        raw_questions = await self._fetch_all_questions(status="active")

        questions = []
        for q in raw_questions:
            # Only include active binary questions with predictions
            if q.get("state") != "active":
                continue
            if q.get("type") != "Forecast::YesNoQuestion":
                continue

            question = self._question_to_model(q)
            if question:
                questions.append(question)

        logger.info(f"Fetched {len(questions)} questions from INFER")
        return questions

    async def fetch_resolution(self, question_id: str) -> Resolution | None:
        """Fetch resolution for a specific question.

        Returns the current crowd forecast probability or resolved value.
        """
        # Fetch the specific question
        questions = await self._fetch_all_questions(status="all", ids=[question_id])
        if not questions:
            return None

        q = questions[0]

        # Get current probability
        answers = q.get("answers", [])
        probability = None
        for answer in answers:
            if answer.get("name", "").lower() == "yes":
                probability = answer.get("probability")
                break

        if probability is None:
            return None

        # Determine resolution date
        if q.get("resolved?"):
            # Use resolved_at or scoring_end_time
            res_dt = parse_iso_datetime(q.get("resolved_at"))
            if not res_dt:
                res_dt = parse_iso_datetime(q.get("scoring_end_time"))
            res_date = res_dt.date() if res_dt else date.today()

            # For resolved questions, probability should be 0 or 1
            # But INFER sometimes has intermediate values, so we use as-is
            return Resolution(
                question_id=question_id,
                source=self.name,
                date=res_date,
                value=probability,
            )

        # Not resolved - return current crowd forecast
        return Resolution(
            question_id=question_id,
            source=self.name,
            date=date.today(),
            value=probability,
        )
