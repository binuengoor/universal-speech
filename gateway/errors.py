"""
Error types shared by the TTS/STT routers.

The distinction matters for the circuit breaker: a bad request (unknown voice,
unsupported format, corrupt upload) is the caller's fault and must never put an
engine into cooldown for everyone else.
"""


class InvalidRequestError(ValueError):
    """The request cannot be served by the selected engine. Maps to HTTP 400."""


class PayloadTooLargeError(InvalidRequestError):
    """The uploaded payload exceeds the configured limit. Maps to HTTP 413."""


def upstream_status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from an upstream client error."""
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "code", None)  # google.api_core exceptions
    if isinstance(code, int):
        return code
    return None


def is_client_error(exc: BaseException) -> bool:
    """True when the failure was caused by the request rather than the engine."""
    if isinstance(exc, InvalidRequestError):
        return True
    code = upstream_status_code(exc)
    return code is not None and 400 <= code < 500 and code not in (401, 403, 408, 429)
