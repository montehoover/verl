EXPECTED_CSRF_TOKEN = 'secure_csrf_token'

def validate_request(headers: dict) -> bool:
    """
    Validates an incoming request by checking the CSRF token.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    token = headers.get('X-CSRF-Token')
    return token == EXPECTED_CSRF_TOKEN

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's information if the user ID exists,
        otherwise None.
    """
    return database["users"].get(user_id)

def change_user_contact(transaction_request: dict) -> bool:
    """
    Updates a user's phone number in the database.

    The transaction_request dictionary should have 'headers' and 'body'.
    The 'body' should contain 'user_id' and 'new_phone_number'.

    Args:
        transaction_request: A dictionary containing request headers and body.

    Returns:
        True if the phone number was updated successfully, False otherwise.
    """
    headers = transaction_request.get("headers")
    if not headers or not validate_request(headers):
        return False

    body = transaction_request.get("body")
    if not body:
        return False

    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    if not user_id or new_phone_number is None: # new_phone_number can be an empty string
        return False

    user_info = get_user_info(user_id)
    if user_info:
        database["users"][user_id]["phone"] = new_phone_number
        return True
    
    return False
