"""Exception handlers mapping LLM failures, which any AI endpoint can hit, to HTTP errors."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import LLMError, LLMResponseError


async def _llm_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """502 for a bad upstream answer, 503 when the LLM is unreachable/unauthorized/missing."""
    code = (
        status.HTTP_502_BAD_GATEWAY
        if isinstance(exc, LLMResponseError)
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(status_code=code, content={"detail": str(exc)})


def register_exception_handlers(app: FastAPI) -> None:
    """Install the LLM error handlers on `app`."""
    app.add_exception_handler(LLMError, _llm_error_handler)
