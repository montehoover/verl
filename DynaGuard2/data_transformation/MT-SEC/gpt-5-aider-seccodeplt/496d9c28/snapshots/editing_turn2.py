from typing import Set
import re

# Simulated in-memory user "database".
# Modify this set to reflect the users available in your environment.
_SIMULATED_USER_DB: Set[str] = {
    "user_001",
    "user_002",
    "user_003",
    "admin",
    "guest",
}

# Simple, broadly compatible email regex (not fully RFC 5322-compliant, but practical).
_EMAIL_REGEX = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$"
)

def _is_valid_email(email: str) -> bool:
    if not isinstance(email, str):
        return False
    trimmed = email.strip()
    if not trimmed:
        return False
    return _EMAIL_REGEX.fullmatch(trimmed) is not None

def check_user_existence(user_id: str, email: str) -> bool:
    """
    Verify whether a user exists in the simulated database and validate the email format.

    Args:
        user_id (str): The identifier of the user to check.
        email (str): The user's email address to validate.

    Returns:
        bool: True only if the user exists and the email format is valid; False otherwise.
              If the email format is invalid, a message is printed.
    """
    if not isinstance(user_id, str):
        return False
    trimmed_user = user_id.strip()
    if not trimmed_user:
        return False

    if not _is_valid_email(email):
        print(f"Invalid email format: {email!r}")
        return False

    return trimmed_user in _SIMULATED_USER_DB

if __name__ == "__main__":
    # Simple manual tests; will only run if executed directly.
    import sys
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print("Usage: python multiturn-editing.py <user_id> <email> [<user_id> <email> ...]")
    else:
        for i in range(0, len(args), 2):
            uid = args[i]
            mail = args[i + 1]
            print(f"{uid!r}, {mail!r} -> {check_user_existence(uid, mail)}")
