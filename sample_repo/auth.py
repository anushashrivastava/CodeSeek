"""Authentication and session handling."""

import hashlib
import time


SESSION_TIMEOUT_SECONDS = 3600


def hash_password(password: str, salt: str) -> str:
    """Combine password and salt, then hash using SHA-256."""
    combined = f"{salt}{password}".encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def check_credentials(username: str, password: str, stored_hash: str, salt: str) -> bool:
    """Verify that a plaintext password matches the stored hash for a user."""
    computed = hash_password(password, salt)
    return computed == stored_hash


def create_session_token(user_id: int) -> str:
    """Generate a simple session token tied to a user id and timestamp."""
    raw = f"{user_id}:{time.time()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def is_session_expired(created_at: float) -> bool:
    """Return True if the session was created longer ago than the timeout."""
    return (time.time() - created_at) > SESSION_TIMEOUT_SECONDS


class LoginManager:
    """Handles login attempts and lockout policy."""

    def __init__(self, max_attempts: int = 5):
        self.max_attempts = max_attempts
        self.failed_attempts = {}

    def record_failure(self, username: str) -> None:
        """Increment the failed login counter for a username."""
        self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1

    def is_locked_out(self, username: str) -> bool:
        """Check whether a user has exceeded the allowed number of failed attempts."""
        return self.failed_attempts.get(username, 0) >= self.max_attempts

    def reset_attempts(self, username: str) -> None:
        """Clear the failed attempt counter after a successful login."""
        self.failed_attempts.pop(username, None)
