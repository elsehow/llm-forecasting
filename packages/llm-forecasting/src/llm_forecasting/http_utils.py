"""Shared HTTP client utilities for sources."""

import httpx


class HTTPClientMixin:
    """Mixin providing lazy HTTP client initialization and cleanup.

    Classes using this mixin should:
    1. Call _init_client(http_client) in __init__ with optional external client
    2. Call await self._get_client() to get a client
    3. Call await self.close() when done

    If an external client is passed in, it won't be closed when close() is called.

    Example:
        class MySource(QuestionSource, HTTPClientMixin):
            def __init__(self, http_client: httpx.AsyncClient | None = None):
                self._init_client(http_client)

            async def fetch_data(self):
                client = await self._get_client()
                response = await client.get(url)
                ...
    """

    _client: httpx.AsyncClient | None = None
    _owns_client: bool = True
    _default_timeout: float = 30.0

    def _init_client(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize client tracking.

        Args:
            http_client: Optional external client. If provided, we won't close it.
            timeout: Custom timeout (only used if creating our own client).
            headers: Custom headers (only used if creating our own client).
        """
        self._client = http_client
        self._owns_client = http_client is None
        if timeout is not None:
            self._default_timeout = timeout
        self._default_headers = headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            kwargs: dict = {"timeout": self._default_timeout}
            if hasattr(self, "_default_headers") and self._default_headers:
                kwargs["headers"] = self._default_headers
            self._client = httpx.AsyncClient(**kwargs)
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
