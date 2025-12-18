"""Multi-tenant RBAC authorization system."""

from scouter.auth.dependencies import (
    get_identity,
    require_all_permissions,
    require_any_permission,
    require_permission,
)
from scouter.auth.exceptions import (
    AuthorizationError,
    InvalidTokenError,
    PermissionDeniedError,
    TenantMembershipError,
    TenantNotFoundError,
    UserNotFoundError,
)
from scouter.auth.middleware import AuthorizationMiddleware
from scouter.auth.rbac import has_all_permissions, has_any_permission, has_permission
from scouter.auth.types import IdentityContext
from scouter.db.auth import (
    build_identity_context,
    create_rbac_constraints,
    get_user_permissions,
    get_user_roles,
    resolve_user_from_oauth,
    verify_tenant_membership,
)

__all__ = [
    "AuthorizationError",
    "AuthorizationMiddleware",
    "IdentityContext",
    "InvalidTokenError",
    "PermissionDeniedError",
    "TenantMembershipError",
    "TenantNotFoundError",
    "UserNotFoundError",
    "build_identity_context",
    "create_rbac_constraints",
    "get_identity",
    "get_user_permissions",
    "get_user_roles",
    "has_all_permissions",
    "has_any_permission",
    "has_permission",
    "require_all_permissions",
    "require_any_permission",
    "require_permission",
    "resolve_user_from_oauth",
    "verify_tenant_membership",
]
