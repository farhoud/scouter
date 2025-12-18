"""FastMCP integration examples for authorization.

Since FastMCP doesn't have built-in middleware, identity context must be
built explicitly by the executor and passed to tools.
"""

from scouter.auth.exceptions import PermissionDeniedError, UserNotFoundError
from scouter.auth.rbac import has_permission
from scouter.auth.types import IdentityContext
from scouter.db.auth import build_identity_context, resolve_user_from_oauth
from scouter.db.neo4j import get_neo4j_driver


def build_identity_from_token(
    token_payload: dict,
    tenant_id: str,
) -> IdentityContext:
    """Build identity context from decoded JWT payload for FastMCP.

    This function should be called by the FastMCP executor after verifying
    the OAuth token externally.

    Args:
        token_payload: Decoded JWT payload
        tenant_id: Tenant identifier (must be provided explicitly)

    Returns:
        IdentityContext dict

    Raises:
        ValueError: If user cannot be resolved or is not in tenant
    """
    provider = token_payload.get("iss")
    sub = token_payload.get("sub")
    if not provider or not sub:
        msg = "Missing provider or sub in token"
        raise ValueError(msg)

    driver = get_neo4j_driver()
    user_id = resolve_user_from_oauth(driver, provider, sub)
    if not user_id:
        msg = "User not found for OAuth identity"
        raise UserNotFoundError(msg)

    return build_identity_context(driver, user_id, tenant_id, token_payload)


def create_user_tool(identity: IdentityContext, payload: dict) -> dict:
    """Example tool that requires user:write permission.

    Args:
        identity: Identity context from build_identity_from_token
        payload: Tool payload

    Returns:
        Tool result

    Raises:
        PermissionError: If permission is denied
    """
    if not has_permission(identity["permissions"], "user:write"):
        msg = "Permission denied: user:write"
        raise PermissionDeniedError(msg)

    # Tool implementation here
    return {"status": "user created", "user_id": payload.get("user_id")}


def list_users_tool(identity: IdentityContext, payload: dict) -> dict:
    """Example tool that requires user:read permission.

    Args:
        identity: Identity context from build_identity_from_token
        payload: Tool payload

    Returns:
        Tool result

    Raises:
        PermissionError: If permission is denied
    """
    if not has_permission(identity["permissions"], "user:read"):
        msg = "Permission denied: user:read"
        raise PermissionDeniedError(msg)

    # Tool implementation here
    return {"users": []}
