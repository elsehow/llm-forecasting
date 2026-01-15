"""Shared HTTP client utilities for sources."""

import httpx


class HTTPClientMixin:
    """Mixin providing lazy HTTP client initialization and cleanup.

    Classes using this mixin should:
    1. Initialize self._client = None (or accept http_client in __init__)
    2. Call await self._get_client() to get a client
    3. Call await self.close() when done

    Example:
        class MySource(QuestionSource, HTTPClientMixin):
            def __init__(self, http_client: httpx.AsyncClient | None = None):
                self._client = http_client

            async def fetch_data(self):
                client = await self._get_client()
                response = await client.get(url)
                ...
    """

    _client: httpx.AsyncClient | None = None
    _default_timeout: float = 30.0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._default_timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client if it exists."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
