"""Pure RBAC authorization functions."""

import fnmatch


def has_permission(permissions: set[str], required: str) -> bool:
    """Check if the given permissions set grants the required permission.

    Rules:
    - '*' grants everything
    - 'resource:*' grants all actions on resource
    - Exact match grants permission

    Args:
        permissions: Set of permission strings
        required: Required permission string

    Returns:
        True if permission is granted, False otherwise
    """
    if "*" in permissions:
        return True

    if required in permissions:
        return True

    # Check wildcard patterns
    return any(fnmatch.fnmatch(required, perm) for perm in permissions)


def has_any_permission(permissions: set[str], required: set[str]) -> bool:
    """Check if any of the required permissions are granted.

    Args:
        permissions: Set of permission strings
        required: Set of required permission strings

    Returns:
        True if at least one permission is granted, False otherwise
    """
    return any(has_permission(permissions, req) for req in required)


def has_all_permissions(permissions: set[str], required: set[str]) -> bool:
    """Check if all required permissions are granted.

    Args:
        permissions: Set of permission strings
        required: Set of required permission strings

    Returns:
        True if all permissions are granted, False otherwise
    """
    return all(has_permission(permissions, req) for req in required)
