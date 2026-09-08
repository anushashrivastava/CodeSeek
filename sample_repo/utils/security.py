"""Low-level security helpers: input sanitization and rate limiting."""

import re
import time
from functools import wraps


def sanitize_input(text: str) -> str:
    """Strip potentially dangerous characters from user-supplied text."""
    return re.sub(r"[<>;'\"]", "", text)


def is_valid_email(email: str) -> bool:
    """Check whether a string looks like a syntactically valid email address."""
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return re.match(pattern, email) is not None


def rate_limit(max_calls: int, period_seconds: float):
    """Decorator factory that limits how often a function can be called."""
    def decorator(func):
        call_times = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            call_times[:] = [t for t in call_times if now - t < period_seconds]
            if len(call_times) >= max_calls:
                raise RuntimeError("rate limit exceeded")
            call_times.append(now)
            return func(*args, **kwargs)

        return wrapper
    return decorator


class InputValidator:
    """Groups related validation rules for incoming request data."""

    def __init__(self, max_length: int = 255):
        self.max_length = max_length

    def validate_length(self, text: str) -> bool:
        """Return True if the text is within the allowed maximum length."""
        return len(text) <= self.max_length

    def validate_username_format(self, username: str) -> bool:
        """Check that a username contains only allowed characters."""
        return bool(re.match(r"^[a-zA-Z0-9_]+$", username))
