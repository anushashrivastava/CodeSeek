"""Database connection and query helpers."""

import sqlite3
from typing import Optional


class Database:
    """Thin wrapper around a sqlite3 connection."""

    def __init__(self, path: str):
        self.path = path
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open a connection to the sqlite database file."""
        self.connection = sqlite3.connect(self.path)

    def close(self) -> None:
        """Close the active database connection, if one exists."""
        if self.connection is not None:
            self.connection.close()

    def execute(self, query: str, params: tuple = ()):
        """Run a SQL query with optional parameters and return the cursor."""
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor

    def commit(self) -> None:
        """Commit the current transaction."""
        self.connection.commit()


def fetch_user_by_id(db: Database, user_id: int):
    """Retrieve a single user row given their numeric id."""
    cursor = db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()


def fetch_user_by_username(db: Database, username: str):
    """Retrieve a single user row given their username."""
    cursor = db.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cursor.fetchone()


def insert_audit_log(db: Database, user_id: int, action: str) -> None:
    """Record an action taken by a user for auditing purposes."""
    db.execute(
        "INSERT INTO audit_log (user_id, action, timestamp) VALUES (?, ?, datetime('now'))",
        (user_id, action),
    )
    db.commit()
