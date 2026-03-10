import hmac
from typing import Dict


# Demo in-memory user credential store.
# Replace with a secure, persistent store that uses salted+hashed passwords.
USER_CREDENTIALS: Dict[str, str] = {
    # Example users (for development/testing only)
    "alice": "correcthorsebatterystaple",
    "bob": "s3cur3!",
}


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user by verifying the provided password.

    Args:
        user_id: The unique identifier for the user.
        password: The plaintext password provided by the user.

    Returns:
        True if the credentials are correct, otherwise False.

    Notes:
        - This implementation uses an in-memory dictionary and compares plaintext
          passwords using constant-time comparison to reduce timing attacks.
        - For production, replace USER_CREDENTIALS with a secure store of
          salted+hashed passwords (e.g., PBKDF2, bcrypt, scrypt, or Argon2).
    """
    expected_password = USER_CREDENTIALS.get(user_id)
    if expected_password is None:
        return False

    # Use constant-time comparison to mitigate timing attacks.
    return hmac.compare_digest(expected_password, password)
