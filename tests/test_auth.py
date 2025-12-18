"""Tests for RBAC authorization system."""

from unittest.mock import Mock

import pytest

from scouter.auth.rbac import has_all_permissions, has_any_permission, has_permission


class TestRBAC:
    """Test RBAC permission checking functions."""

    def test_has_permission_exact_match(self):
        """Test exact permission match."""
        permissions = {"user:read", "user:write"}
        assert has_permission(permissions, "user:read") is True
        assert has_permission(permissions, "user:delete") is False

    def test_has_permission_wildcard_all(self):
        """Test wildcard '*' grants everything."""
        permissions = {"*"}
        assert has_permission(permissions, "user:read") is True
        assert has_permission(permissions, "billing:invoice:create") is True

    def test_has_permission_resource_wildcard(self):
        """Test resource:* grants all actions on resource."""
        permissions = {"user:*", "billing:read"}
        assert has_permission(permissions, "user:read") is True
        assert has_permission(permissions, "user:write") is True
        assert has_permission(permissions, "billing:read") is True
        assert has_permission(permissions, "billing:write") is False

    def test_has_any_permission(self):
        """Test has_any_permission function."""
        permissions = {"user:read", "billing:write"}
        assert has_any_permission(permissions, {"user:read", "user:write"}) is True
        assert has_any_permission(permissions, {"user:write", "billing:read"}) is False

    def test_has_all_permissions(self):
        """Test has_all_permissions function."""
        permissions = {"user:read", "user:write", "billing:read"}
        assert has_all_permissions(permissions, {"user:read", "billing:read"}) is True
        assert has_all_permissions(permissions, {"user:read", "billing:write"}) is False


class TestNeo4jQueries:
    """Test Neo4j query functions."""

    @pytest.fixture
    def mock_driver(self):
        """Mock Neo4j driver."""
        return Mock()

    @pytest.fixture
    def mock_session(self, mock_driver):
        """Mock Neo4j session."""
        session = Mock()
        mock_driver.session.return_value.__enter__.return_value = session
        return session

    def test_get_user_permissions(self, mock_driver, mock_session):
        """Test getting user permissions."""
        from scouter.db.auth import get_user_permissions

        mock_session.run.return_value = [
            Mock(data=lambda: {"permission": "user:read"}),
            Mock(data=lambda: {"permission": "user:write"}),
        ]

        result = get_user_permissions(mock_driver, "user1", "tenant1")
        assert result == {"user:read", "user:write"}

    def test_resolve_user_from_oauth(self, mock_driver, mock_session):
        """Test resolving user from OAuth identity."""
        from scouter.db.auth import resolve_user_from_oauth

        mock_session.run.return_value.single.return_value = {"user_id": "user1"}

        result = resolve_user_from_oauth(mock_driver, "auth0", "sub123")
        assert result == "user1"

    def test_verify_tenant_membership(self, mock_driver, mock_session):
        """Test verifying tenant membership."""
        from scouter.db.auth import verify_tenant_membership

        mock_session.run.return_value.single.return_value = {"is_member": True}

        result = verify_tenant_membership(mock_driver, "user1", "tenant1")
        assert result is True
