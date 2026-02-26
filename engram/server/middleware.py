"""Server middleware for request logging and error handling."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs request method, path, and duration."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.monotonic()
        logger.info(
            "[%s] %s %s", request_id, request.method, request.url.path
        )

        response = await call_next(request)

        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "[%s] %s %s → %d (%.1fms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
