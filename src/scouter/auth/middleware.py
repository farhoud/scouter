"""FastAPI middleware for identity resolution and authorization."""

import jwt
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from scouter.db.auth import build_identity_context, resolve_user_from_oauth
from scouter.db.neo4j import get_neo4j_driver


class AuthorizationMiddleware:
    """Middleware for building identity context from OAuth tokens."""

    def __init__(self, jwks_url: str, issuer: str, audience: str):
        """Initialize middleware with OAuth configuration.

        Args:
            jwks_url: URL to fetch JWKS for token verification
            issuer: Expected token issuer
            audience: Expected token audience
        """
        self.jwks_url = jwks_url
        self.issuer = issuer
        self.audience = audience
        self.jwks_client = jwt.PyJWKClient(jwks_url)

    async def __call__(self, request: Request, call_next) -> Response:
        """Process request and build identity context."""
        # Extract bearer token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid authorization header"},
            )

        token = auth_header[7:]  # Remove "Bearer "

        try:
            # Verify token
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=self.audience,
            )

            # Extract user identity
            provider = payload.get("iss")  # Assuming issuer indicates provider
            sub = payload.get("sub")
            if not provider or not sub:
                msg = "Missing provider or sub in token"
                raise ValueError(msg)

            # Resolve user from OAuth identity
            driver = get_neo4j_driver()
            user_id = resolve_user_from_oauth(driver, provider, sub)
            if not user_id:
                msg = "User not found for OAuth identity"
                raise ValueError(msg)

            # Resolve tenant
            tenant_id = self._resolve_tenant_id(request, payload)
            if not tenant_id:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Tenant not specified or invalid"},
                )

            # Build identity context
            identity = build_identity_context(driver, user_id, tenant_id, payload)

            # Attach to request state
            request.state.identity = identity

        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token has expired"},
            )
        except jwt.InvalidTokenError as e:
            return JSONResponse(
                status_code=401,
                content={"detail": f"Invalid token: {e!s}"},
            )
        except ValueError as e:
            return JSONResponse(
                status_code=403,
                content={"detail": str(e)},
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": f"Authorization error: {e!s}"},
            )

        # Continue with request
        return await call_next(request)

    def _resolve_tenant_id(self, request: Request, payload: dict) -> str | None:
        """Resolve tenant ID from request or token.

        Priority:
        1. JWT claim (tenant_id)
        2. Request header (X-Tenant-ID)

        Args:
            request: FastAPI request object
            payload: Decoded JWT payload

        Returns:
            Tenant ID if found, None otherwise
        """
        # Check JWT claim first
        tenant_id = payload.get("tenant_id")
        if tenant_id:
            return tenant_id

        # Check header
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            return tenant_id

        return None
