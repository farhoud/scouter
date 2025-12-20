"""FastAPI dependencies for authorization."""

from fastapi import Depends, HTTPException, Request

from scouter.auth.rbac import has_permission
from scouter.auth.types import IdentityContext


def get_identity(request: Request) -> IdentityContext:
    """Dependency to get identity context from request state.

    Args:
        request: FastAPI request object

    Returns:
        IdentityContext

    Raises:
        HTTPException: If identity is not found in request state
    """
    identity = getattr(request.state, "identity", None)
    if not identity:
        raise HTTPException(
            status_code=401,
            detail="Identity context not found. Ensure authorization middleware is configured.",
        )
    return identity


def require_permission(required: str):
    """Create a dependency that requires a specific permission.

    Args:
        required: Permission string that must be granted

    Returns:
        Dependency function that checks permission
    """

    def dependency(
        identity: IdentityContext = Depends(get_identity),
    ) -> IdentityContext:
        if not has_permission(identity.permissions, required):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {required}",
            )
        return identity

    return dependency


def require_any_permission(required: set[str]):
    """Create a dependency that requires any of the specified permissions.

    Args:
        required: Set of permission strings, at least one must be granted

    Returns:
        Dependency function that checks permissions
    """

    def dependency(
        identity: IdentityContext = Depends(get_identity),
    ) -> IdentityContext:
        if not any(has_permission(identity.permissions, perm) for perm in required):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: requires one of {required}",
            )
        return identity

    return dependency


def require_all_permissions(required: set[str]):
    """Create a dependency that requires all specified permissions.

    Args:
        required: Set of permission strings, all must be granted

    Returns:
        Dependency function that checks permissions
    """

    def dependency(
        identity: IdentityContext = Depends(get_identity),
    ) -> IdentityContext:
        if not all(has_permission(identity.permissions, perm) for perm in required):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: requires all of {required}",
            )
        return identity

    return dependency
