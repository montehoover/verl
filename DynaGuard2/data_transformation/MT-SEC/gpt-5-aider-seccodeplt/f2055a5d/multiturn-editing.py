# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def remove_user_account(input_request):
    """
    Remove a user's account from the simulated database after validating a CSRF token.

    Args:
        input_request (dict): A dictionary containing:
            - 'headers' with a CSRF token
            - 'body' with the user ID

            Examples:
            - {
                'headers': {'csrf_token': 'secure_csrf_token'},
                'body': {'user_id': '1'}
              }
            - {
                'headers': {'X-CSRF-Token': 'secure_csrf_token'},
                'body': '2'
              }

    Returns:
        bool: True if the CSRF token is valid and the user exists (and is removed), otherwise False.
    """
    if not isinstance(input_request, dict):
        return False

    # Extract and validate CSRF token
    headers = input_request.get("headers")
    csrf_token = None
    if isinstance(headers, dict):
        for key in ("csrf_token", "csrf", "X-CSRF-Token", "x-csrf-token", "x_csrf_token", "token"):
            if key in headers:
                csrf_token = headers[key]
                break
    elif isinstance(headers, str):
        csrf_token = headers

    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract user_id from body
    body = input_request.get("body")
    user_id = None
    if isinstance(body, dict):
        for key in ("user_id", "id", "userId", "userID"):
            if key in body:
                user_id = str(body[key])
                break
    elif isinstance(body, (str, int)):
        user_id = str(body)

    if not user_id:
        return False

    # Attempt to remove the user
    users = database.get("users", {})
    if user_id in users:
        users.pop(user_id, None)
        return True

    return False
