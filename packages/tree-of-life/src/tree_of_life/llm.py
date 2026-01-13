"""LLM utilities for the pipeline with batch API support."""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import BaseModel

from .config import MODEL, USE_BATCH_API, STRUCTURED_OUTPUTS_BETA
from .schemas import make_strict_schema

# Configure logging
logger = logging.getLogger(__name__)

# Anthropic batch API custom_id constraints (re used here)
CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class BatchValidationError(ValueError):
    """Raised when batch request validation fails."""
    pass


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


def validate_custom_id(custom_id: str) -> None:
    """Validate custom_id matches Anthropic batch API requirements.

    Must match pattern: ^[a-zA-Z0-9_-]{1,64}$
    """
    if not CUSTOM_ID_PATTERN.match(custom_id):
        raise BatchValidationError(
            f"Invalid custom_id '{custom_id}'. "
            f"Must match pattern ^[a-zA-Z0-9_-]{{1,64}}$ "
            f"(only alphanumeric, underscore, hyphen; max 64 chars)"
        )


@dataclass
class LLMRequest:
    """A single LLM request."""
    custom_id: str
    system: str
    user: str
    response_schema: dict | None = field(default=None)

    def __post_init__(self):
        """Validate custom_id on creation."""
        validate_custom_id(self.custom_id)


def get_client() -> anthropic.Anthropic:
    """Get Anthropic client."""
    return anthropic.Anthropic()


def get_model_id(model: str) -> str:
    """Convert model string to Anthropic model ID."""
    if model.startswith("anthropic/"):
        return model.replace("anthropic/", "")
    return model


async def llm_call_sync(
    system: str,
    user: str,
    model: str,
    response_model: type[BaseModel],
) -> dict:
    """Make a single synchronous LLM call with structured outputs.

    Args:
        system: System prompt
        user: User prompt
        model: Model to use
        response_model: Pydantic model for structured outputs.
                       Uses Claude's structured outputs feature
                       to guarantee valid JSON matching the schema.
    """
    client = get_client()
    model_id = get_model_id(model)

    logger.debug(f"Making sync LLM call to {model_id}")

    schema = make_strict_schema(response_model)

    try:
        response = client.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            extra_headers={"anthropic-beta": STRUCTURED_OUTPUTS_BETA},
            extra_body={
                "output_format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        )
        content = response.content[0].text
        logger.debug(f"Received response: {len(content)} chars")

        # Structured outputs guarantee valid JSON
        return json.loads(content)
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        raise LLMError(f"API call failed: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        raise LLMError(f"Failed to parse LLM response as JSON: {e}") from e


async def llm_call(
    system: str,
    user: str,
    model: str,
    response_model: type[BaseModel],
) -> dict:
    """Make an LLM call with structured outputs (uses sync mode for single calls)."""
    return await llm_call_sync(system, user, model, response_model)


def create_batch(requests: list[LLMRequest], model: str = MODEL) -> str:
    """Create a batch of requests and return batch ID.

    If any request has a response_schema, structured outputs will be enabled
    for that request using the Claude structured outputs beta.
    """
    client = get_client()
    model_id = get_model_id(model)

    # Check if any requests use structured outputs
    uses_structured = any(req.response_schema is not None for req in requests)

    logger.info(f"Creating batch with {len(requests)} requests using {model_id}")
    if uses_structured:
        logger.info(f"  Using structured outputs beta: {STRUCTURED_OUTPUTS_BETA}")

    batch_requests = []
    for i, req in enumerate(requests):
        logger.debug(f"  Request {i}: custom_id={req.custom_id}")
        params = {
            "model": model_id,
            "max_tokens": 4096,
            "system": req.system,
            "messages": [{"role": "user", "content": req.user}],
        }

        # Add structured output format if schema provided
        if req.response_schema is not None:
            params["output_format"] = {
                "type": "json_schema",
                "schema": req.response_schema,
            }

        batch_requests.append({
            "custom_id": req.custom_id,
            "params": params,
        })

    try:
        # Add beta header if any request uses structured outputs
        create_kwargs: dict[str, Any] = {"requests": batch_requests}
        if uses_structured:
            create_kwargs["extra_headers"] = {"anthropic-beta": STRUCTURED_OUTPUTS_BETA}

        batch = client.messages.batches.create(**create_kwargs)
        logger.info(f"Created batch: {batch.id}")
        return batch.id
    except anthropic.BadRequestError as e:
        logger.error(f"Batch creation failed: {e}")
        # Log which custom_ids were in the batch for debugging
        custom_ids = [r.custom_id for r in requests]
        logger.error(f"Custom IDs in failed batch: {custom_ids[:5]}... (showing first 5)")
        raise LLMError(
            f"Batch creation failed: {e}\n"
            f"Custom IDs: {custom_ids[:5]}..."
        ) from e
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error during batch creation: {e}")
        raise LLMError(f"Batch creation failed: {e}") from e


def poll_batch(batch_id: str, poll_interval: float = 30.0, verbose: bool = True) -> dict[str, dict]:
    """Poll batch until complete, return results keyed by custom_id."""
    client = get_client()
    logger.info(f"Polling batch {batch_id}")

    while True:
        batch = client.messages.batches.retrieve(batch_id)

        status_msg = (
            f"status={batch.processing_status} "
            f"(succeeded={batch.request_counts.succeeded}, "
            f"processing={batch.request_counts.processing}, "
            f"errored={batch.request_counts.errored})"
        )
        logger.debug(f"Batch {batch_id}: {status_msg}")

        if verbose:
            print(f"  Batch {batch_id[:12]}... status={batch.processing_status} "
                  f"({batch.request_counts.succeeded}/{batch.request_counts.processing}/"
                  f"{batch.request_counts.errored})")

        if batch.processing_status == "ended":
            break

        time.sleep(poll_interval)

    # Log final stats
    logger.info(
        f"Batch {batch_id} completed: "
        f"{batch.request_counts.succeeded} succeeded, "
        f"{batch.request_counts.errored} errored"
    )

    # Retrieve results - structured outputs guarantee valid JSON
    results = {}
    errors = []
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            content = result.result.message.content[0].text
            try:
                results[custom_id] = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error for {custom_id}: {e}")
                errors.append((custom_id, str(e)))
                results[custom_id] = {"error": f"JSON parse error: {e}"}
        else:
            error_msg = str(result.result)
            logger.error(f"Batch item {custom_id} failed: {error_msg}")
            errors.append((custom_id, error_msg))
            results[custom_id] = {"error": error_msg}

    if errors:
        logger.warning(f"Batch had {len(errors)} errors: {errors[:3]}...")

    return results


async def llm_batch(
    requests: list[LLMRequest],
    model: str = MODEL,
    poll_interval: float = 30.0,
    verbose: bool = True,
) -> dict[str, dict]:
    """Submit batch and wait for results."""
    if verbose:
        print(f"  Submitting batch of {len(requests)} requests...")

    batch_id = create_batch(requests, model)

    if verbose:
        print(f"  Batch ID: {batch_id}")

    # Poll in a thread to not block
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, lambda: poll_batch(batch_id, poll_interval, verbose)
    )

    return results


async def llm_call_many(
    calls: list[tuple[str, str, str]],  # [(custom_id, system, user), ...]
    model: str,
    response_model: type[BaseModel],
    use_batch: bool = USE_BATCH_API,
    verbose: bool = True,
) -> dict[str, dict]:
    """Make multiple LLM calls with structured outputs.

    Args:
        calls: List of (custom_id, system, user) tuples
        model: Model to use
        response_model: Pydantic model for structured outputs.
                       All calls will use this schema.
        use_batch: Whether to use batch API (default from config)
        verbose: Whether to print progress

    Returns:
        Dict mapping custom_id -> parsed JSON response.

    Note: custom_id must match ^[a-zA-Z0-9_-]{1,64}$ for batch API.
    Use underscores or hyphens as separators, not colons or other characters.
    """
    logger.info(f"llm_call_many: {len(calls)} calls, batch={use_batch}")

    if not use_batch:
        # Sync mode: parallel async calls
        if verbose:
            print(f"  Making {len(calls)} parallel sync calls...")

        async def call_with_id(custom_id: str, system: str, user: str) -> tuple[str, dict]:
            result = await llm_call_sync(system, user, model, response_model)
            return custom_id, result

        tasks = [call_with_id(cid, sys, usr) for cid, sys, usr in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        output = {}
        for i, result in enumerate(results):
            custom_id = calls[i][0]
            if isinstance(result, Exception):
                logger.error(f"Call {custom_id} failed: {result}")
                output[custom_id] = {"error": str(result)}
            else:
                output[custom_id] = result[1]
        return output

    # Batch mode - validate all custom_ids upfront
    for cid, _, _ in calls:
        validate_custom_id(cid)  # Raises BatchValidationError if invalid

    # Create requests with schema
    response_schema = make_strict_schema(response_model)
    requests = [
        LLMRequest(custom_id=cid, system=sys, user=usr, response_schema=response_schema)
        for cid, sys, usr in calls
    ]
    return await llm_batch(requests, model, verbose=verbose)
