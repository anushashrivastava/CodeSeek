"""User account management: creation, updates, and profile logic."""

from database import Database, fetch_user_by_username
from auth import hash_password


class User:
    """Represents a registered user account."""

    def __init__(self, user_id: int, username: str, email: str):
        self.user_id = user_id
        self.username = username
        self.email = email

    def display_name(self) -> str:
        """Return a human-friendly name for display purposes."""
        return self.username.capitalize()

    def to_dict(self) -> dict:
        """Serialize the user to a plain dictionary."""
        return {"id": self.user_id, "username": self.username, "email": self.email}


def register_user(db: Database, username: str, password: str, email: str) -> User:
    """Create a new user account with a securely hashed password."""
    salt = "static_salt_for_demo"
    password_hash = hash_password(password, salt)
    db.execute(
        "INSERT INTO users (username, password_hash, salt, email) VALUES (?, ?, ?, ?)",
        (username, password_hash, salt, email),
    )
    db.commit()
    return User(user_id=-1, username=username, email=email)


def deactivate_account(db: Database, username: str) -> bool:
    """Mark a user's account as inactive rather than deleting their data."""
    row = fetch_user_by_username(db, username)
    if row is None:
        return False
    db.execute("UPDATE users SET is_active = 0 WHERE username = ?", (username,))
    db.commit()
    return True


def update_email(db: Database, username: str, new_email: str) -> None:
    """Change the email address on file for a given user."""
    db.execute("UPDATE users SET email = ? WHERE username = ?", (new_email, username))
    db.commit()
