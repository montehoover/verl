# In a real application, this token would be securely generated and managed,
# for example, per session. For this example, we'll use a hardcoded token.
KNOWN_CSRF_TOKEN = "supersecrettoken123"
CSRF_HEADER_NAME = "X-CSRF-Token"

def validate_csrf_token(headers: dict) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    token_from_request = headers.get(CSRF_HEADER_NAME)
    if not token_from_request:
        return False
    
    return token_from_request == KNOWN_CSRF_TOKEN

if __name__ == '__main__':
    # Example Usage
    valid_headers = {
        "Content-Type": "application/json",
        CSRF_HEADER_NAME: "supersecrettoken123"
    }
    invalid_headers_wrong_token = {
        "Content-Type": "application/json",
        CSRF_HEADER_NAME: "wrongtoken"
    }
    invalid_headers_missing_token = {
        "Content-Type": "application/json"
    }

    print(f"Validation with correct token: {validate_csrf_token(valid_headers)}")
    print(f"Validation with wrong token: {validate_csrf_token(invalid_headers_wrong_token)}")
    print(f"Validation with missing token: {validate_csrf_token(invalid_headers_missing_token)}")
