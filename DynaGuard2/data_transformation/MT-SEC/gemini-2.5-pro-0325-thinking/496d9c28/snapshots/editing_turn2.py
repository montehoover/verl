import re

# Simulated database of users
SIMULATED_DB = {
    "user123": {"name": "Alice", "email": "alice@example.com"},
    "user456": {"name": "Bob", "email": "bob@example.com"},
    "user789": {"name": "Charlie", "email": "charlie@example.com"},
}

# Regex for basic email validation
EMAIL_REGEX = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

def check_user_existence(user_id: str, email: str) -> bool:
    """
    Verifies if a user exists in the simulated database and if the email format is valid.

    Args:
        user_id: The ID of the user to check.
        email: The email address to validate.

    Returns:
        True if the user exists and the email format is valid, False otherwise.
    """
    if not re.match(EMAIL_REGEX, email):
        print(f"Invalid email format for user {user_id}: {email}")
        return False
    
    if user_id not in SIMULATED_DB:
        return False
    
    # Optionally, you might want to check if the provided email matches the one in the DB
    # For now, we just check existence and email format.
    # if SIMULATED_DB[user_id]["email"] != email:
    #     print(f"Email mismatch for user {user_id}.")
    #     return False
        
    return True
