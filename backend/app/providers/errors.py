class ProviderError(Exception):
    """A known upstream failure that can safely be returned to API clients."""

    status_code = 502
    error_type = "provider_error"
    code = "provider_error"

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class ProviderTimeoutError(ProviderError):
    status_code = 504
    error_type = "provider_timeout"
    code = "upstream_timeout"


class ProviderAuthenticationError(ProviderError):
    status_code = 502
    error_type = "provider_authentication_error"
    code = "upstream_authentication_failed"


class ProviderUnavailableError(ProviderError):
    status_code = 503
    error_type = "provider_unavailable"
    code = "upstream_unavailable"


class ProviderRateLimitError(ProviderUnavailableError):
    status_code = 429
    error_type = "provider_rate_limited"
    code = "upstream_rate_limited"


class MalformedProviderResponseError(ProviderError):
    status_code = 502
    error_type = "malformed_provider_response"
    code = "invalid_upstream_response"


class ModelNotAvailableError(ProviderError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "model_not_available"
