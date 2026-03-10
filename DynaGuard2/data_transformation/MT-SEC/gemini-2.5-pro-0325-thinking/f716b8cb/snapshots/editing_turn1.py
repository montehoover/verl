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

if __name__ == '__main__':
    # Example Usage
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
