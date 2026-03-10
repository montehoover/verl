# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(request_headers: dict) -> bool:
    """
    Validates the CSRF token from request headers.

    Args:
        request_headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    # Common header name for CSRF tokens is 'X-CSRF-Token'
    # or 'X-XSRF-TOKEN' depending on the framework.
    # We'll use 'X-CSRF-Token' for this example.
    actual_token = request_headers.get('X-CSRF-Token')

    return actual_token == VALID_CSRF_TOKEN

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user details if found, otherwise None.
    """
    return user_database["users"].get(user_id)

def change_user_email(new_request: dict) -> bool:
    """
    Updates a user's email address securely.

    Args:
        new_request: A dictionary containing headers and a body with user_id and new_email.
                     Example: {'headers': {'X-CSRF-Token': '...'}, 
                               'body': {'user_id': '1', 'new_email': 'new.email@example.com'}}

    Returns:
        True if the update is successful, False otherwise.
    """
    headers = new_request.get('headers', {})
    body = new_request.get('body', {})

    user_id = body.get('user_id')
    new_email = body.get('new_email')

    if not user_id or not new_email:
        return False  # Essential data missing

    # Validate CSRF token
    if not validate_csrf_token(headers):
        return False

    # Check if user exists
    user_info = get_user_info(user_id)
    if user_info is None:
        return False

    # Update email in the database
    # Ensure the user_id exists in the "users" dictionary before trying to update
    if user_id in user_database["users"]:
        user_database["users"][user_id]["email"] = new_email
        return True
    else:
        # This case should ideally be caught by get_user_info returning None,
        # but as a safeguard:
        return False
