"""Authorization types and data structures."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IdentityContext:
    """Request-scoped identity context containing user and authorization information."""

    user_id: str
    """Unique user identifier"""

    tenant_id: str
    """Tenant identifier for multi-tenancy"""

    roles: set[str]
    """Set of role names assigned to the user in the tenant"""

    permissions: set[str]
    """Set of permission keys granted to the user"""

    token_claims: dict[str, Any]
    """Raw JWT token claims for additional context"""

    extra: dict[str, Any] = field(default_factory=dict)
    """Additional extra information"""
