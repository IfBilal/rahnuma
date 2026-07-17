"""Clerk session-token verification for FastAPI routes."""

import os

from clerk_backend_api import AuthenticateRequestOptions, authenticate_request
from fastapi import HTTPException, Request, status


def get_current_user(request: Request) -> dict[str, str]:
    """Verify the Clerk Bearer token and return its stable Clerk user ID.

    Clerk owns credentials, sessions, and token issuance. Rahnuma only accepts
    a verified session token and uses its `sub` claim as the profile owner ID.
    The SDK fetches and caches Clerk's JWKS when CLERK_JWT_KEY is not supplied,
    so the optional PEM key is not required for local development.
    """
    state = authenticate_request(
        request,
        AuthenticateRequestOptions(
            secret_key=os.environ["CLERK_SECRET_KEY"],
            jwt_key=os.environ.get("CLERK_JWT_KEY") or None,
            authorized_parties=["http://localhost:3000"],
            accepts_token=["session_token"],
        ),
    )
    if not state.is_signed_in or not state.payload or not state.payload.get("sub"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sign in to continue.")
    return {"id": state.payload["sub"]}
