"""Authorization types and data structures."""

from typing import Any

IdentityContext = dict[str, Any]
"""Request-scoped identity context containing user and authorization information.

Keys:
    user_id: str - Unique user identifier
    tenant_id: str - Tenant identifier for multi-tenancy
    roles: set[str] - Set of role names assigned to the user in the tenant
    permissions: set[str] - Set of permission keys granted to the user
    token_claims: dict - Raw JWT token claims for additional context
"""
