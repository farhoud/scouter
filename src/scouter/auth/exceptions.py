"""Authorization-specific exceptions."""


class AuthorizationError(Exception):
    """Base exception for authorization errors."""


class InvalidTokenError(AuthorizationError):
    """Raised when OAuth token is invalid or expired."""


class UserNotFoundError(AuthorizationError):
    """Raised when user cannot be resolved from OAuth identity."""


class TenantNotFoundError(AuthorizationError):
    """Raised when tenant is not specified or invalid."""


class PermissionDeniedError(AuthorizationError):
    """Raised when user lacks required permissions."""


class TenantMembershipError(AuthorizationError):
    """Raised when user is not a member of the required tenant."""
