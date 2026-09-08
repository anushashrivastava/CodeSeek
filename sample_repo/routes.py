"""HTTP route handlers for the (hypothetical) web application."""

from database import Database, fetch_user_by_username
from auth import check_credentials, create_session_token, LoginManager
from users import register_user


login_manager = LoginManager()


def handle_login_request(db: Database, username: str, password: str):
    """Process a login attempt: validate credentials and issue a session token."""
    if login_manager.is_locked_out(username):
        return {"error": "account locked"}, 403

    row = fetch_user_by_username(db, username)
    if row is None:
        login_manager.record_failure(username)
        return {"error": "invalid credentials"}, 401

    stored_hash, salt = row["password_hash"], row["salt"]
    if not check_credentials(username, password, stored_hash, salt):
        login_manager.record_failure(username)
        return {"error": "invalid credentials"}, 401

    login_manager.reset_attempts(username)
    token = create_session_token(row["id"])
    return {"token": token}, 200


def handle_signup_request(db: Database, username: str, password: str, email: str):
    """Process a new-account signup request."""
    existing = fetch_user_by_username(db, username)
    if existing is not None:
        return {"error": "username taken"}, 409
    user = register_user(db, username, password, email)
    return user.to_dict(), 201


def handle_logout_request(session_token: str):
    """Invalidate a session token, logging the user out."""
    return {"status": "logged out"}, 200
