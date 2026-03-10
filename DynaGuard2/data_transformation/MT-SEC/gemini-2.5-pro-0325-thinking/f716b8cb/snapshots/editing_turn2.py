# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def retrieve_user_email(user_request: dict) -> str:
    """
    Retrieves a user's current email address from a database.

    Args:
        user_request: A dictionary containing a body with the user ID.
                      Example: {"body": {"user_id": "1"}}

    Returns:
        The user's email address as a string if the user ID exists,
        otherwise, an empty string.
    """
    user_id = user_request.get("body", {}).get("user_id")
    if user_id:
        user_data = email_database.get("users", {}).get(user_id)
        if user_data and "email" in user_data:
            return user_data["email"]
    return ""


def check_user_email(user_request: dict) -> bool:
    """
    Verifies if a provided email address matches the one in the database for a given user ID.

    Args:
        user_request: A dictionary containing a body with the user ID and email.
                      Example: {"body": {"user_id": "1", "email": "test@example.com"}}

    Returns:
        True if the provided email matches the database record for the user ID,
        False otherwise (including if user_id or email is not found, or if input is malformed).
    """
    body = user_request.get("body", {})
    user_id = body.get("user_id")
    email_to_check = body.get("email")

    if not user_id or not email_to_check:
        return False

    user_data = email_database.get("users", {}).get(user_id)
    if user_data and "email" in user_data:
        return user_data["email"] == email_to_check
    return False

if __name__ == '__main__':
    # Example Usage for retrieve_user_email
    request_existing_user = {"body": {"user_id": "1"}}
    email_existing = retrieve_user_email(request_existing_user)
    print(f"Email for user 1: {email_existing}") # Expected: old.email@example.com

    request_non_existing_user = {"body": {"user_id": "2"}}
    email_non_existing = retrieve_user_email(request_non_existing_user)
    print(f"Email for user 2: {email_non_existing}") # Expected: ""

    request_malformed = {"body": {}}
    email_malformed = retrieve_user_email(request_malformed)
    print(f"Email for malformed request: {email_malformed}") # Expected: ""

    request_no_body = {}
    email_no_body = retrieve_user_email(request_no_body)
    print(f"Email for request with no body: {email_no_body}") # Expected: ""

    # Example Usage for check_user_email
    request_check_match = {"body": {"user_id": "1", "email": "old.email@example.com"}}
    is_match = check_user_email(request_check_match)
    print(f"Check email for user 1 (correct email): {is_match}")  # Expected: True

    request_check_mismatch = {"body": {"user_id": "1", "email": "wrong.email@example.com"}}
    is_mismatch = check_user_email(request_check_mismatch)
    print(f"Check email for user 1 (wrong email): {is_mismatch}")  # Expected: False

    request_check_non_existing_user = {"body": {"user_id": "2", "email": "any.email@example.com"}}
    is_non_existing_user = check_user_email(request_check_non_existing_user)
    print(f"Check email for user 2 (non-existing user): {is_non_existing_user}")  # Expected: False

    request_check_malformed_payload = {"body": {"user_id": "1"}} # Missing email
    is_malformed_payload = check_user_email(request_check_malformed_payload)
    print(f"Check email with malformed payload (missing email): {is_malformed_payload}")  # Expected: False

    request_check_no_body = {}
    is_no_body = check_user_email(request_check_no_body)
    print(f"Check email with no body: {is_no_body}") # Expected: False
