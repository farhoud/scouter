"""Neo4j operations for authorization and RBAC."""

from typing import Any

import neo4j
from scouter.auth.types import IdentityContext


def create_rbac_constraints(driver: neo4j.Driver) -> None:
    """Create Neo4j constraints and indexes for RBAC model.

    This should be run once during database setup.

    Args:
        driver: Neo4j driver instance
    """
    constraints = [
        "CREATE CONSTRAINT tenant_id_unique IF NOT EXISTS FOR (t:Tenant) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
        "CREATE CONSTRAINT oauth_identity_unique IF NOT EXISTS FOR (oi:OAuthIdentity) REQUIRE (oi.provider, oi.sub) IS UNIQUE",
        "CREATE CONSTRAINT role_id_unique IF NOT EXISTS FOR (r:Role) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT permission_key_unique IF NOT EXISTS FOR (p:Permission) REQUIRE p.key IS UNIQUE",
    ]

    with driver.session() as session:
        for constraint in constraints:
            session.run(constraint)


def get_user_permissions(
    driver: neo4j.Driver, user_id: str, tenant_id: str
) -> set[str]:
    """Get all permissions for a user within a specific tenant.

    Args:
        driver: Neo4j driver instance
        user_id: User identifier
        tenant_id: Tenant identifier

    Returns:
        Set of permission keys granted to the user in the tenant
    """
    query = """
    MATCH (u:User {id: $user_id})-[:MEMBER_OF]->(t:Tenant {id: $tenant_id})
    MATCH (u)-[:HAS_ROLE]->(r:Role)-[:ROLE_IN]->(t)
    MATCH (r)-[:GRANTS]->(p:Permission)
    RETURN DISTINCT p.key as permission
    """
    with driver.session() as session:
        result = session.run(query, user_id=user_id, tenant_id=tenant_id)
        return {record["permission"] for record in result}


def get_user_roles(driver: neo4j.Driver, user_id: str, tenant_id: str) -> set[str]:
    """Get all roles for a user within a specific tenant.

    Args:
        driver: Neo4j driver instance
        user_id: User identifier
        tenant_id: Tenant identifier

    Returns:
        Set of role names assigned to the user in the tenant
    """
    query = """
    MATCH (u:User {id: $user_id})-[:MEMBER_OF]->(t:Tenant {id: $tenant_id})
    MATCH (u)-[:HAS_ROLE]->(r:Role)-[:ROLE_IN]->(t)
    RETURN DISTINCT r.name as role
    """
    with driver.session() as session:
        result = session.run(query, user_id=user_id, tenant_id=tenant_id)
        return {record["role"] for record in result}


def resolve_user_from_oauth(
    driver: neo4j.Driver, provider: str, sub: str
) -> str | None:
    """Resolve user ID from OAuth identity.

    Args:
        driver: Neo4j driver instance
        provider: OAuth provider name
        sub: OAuth subject identifier

    Returns:
        User ID if found, None otherwise
    """
    query = """
    MATCH (oi:OAuthIdentity {provider: $provider, sub: $sub})-[:IDENTIFIES]->(u:User)
    RETURN u.id as user_id
    """
    with driver.session() as session:
        result = session.run(query, provider=provider, sub=sub)
        record = result.single()
        return record["user_id"] if record else None


def verify_tenant_membership(
    driver: neo4j.Driver, user_id: str, tenant_id: str
) -> bool:
    """Verify that a user is a member of a tenant.

    Args:
        driver: Neo4j driver instance
        user_id: User identifier
        tenant_id: Tenant identifier

    Returns:
        True if user is member of tenant, False otherwise
    """
    query = """
    MATCH (u:User {id: $user_id})-[:MEMBER_OF]->(t:Tenant {id: $tenant_id})
    RETURN count(t) > 0 as is_member
    """
    with driver.session() as session:
        result = session.run(query, user_id=user_id, tenant_id=tenant_id)
        record = result.single()
        return record["is_member"] if record else False


def build_identity_context(
    driver: neo4j.Driver,
    user_id: str,
    tenant_id: str,
    token_claims: dict[str, Any],
) -> IdentityContext:
    """Build complete identity context for a user and tenant.

    Args:
        driver: Neo4j driver instance
        user_id: User identifier
        tenant_id: Tenant identifier
        token_claims: Raw JWT token claims

    Returns:
        Complete IdentityContext

    Raises:
        ValueError: If user is not a member of the tenant
    """
    if not verify_tenant_membership(driver, user_id, tenant_id):
        msg = f"User {user_id} is not a member of tenant {tenant_id}"
        raise ValueError(msg)

    roles = get_user_roles(driver, user_id, tenant_id)
    permissions = get_user_permissions(driver, user_id, tenant_id)

    return IdentityContext(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=roles,
        permissions=permissions,
        token_claims=token_claims,
    )
