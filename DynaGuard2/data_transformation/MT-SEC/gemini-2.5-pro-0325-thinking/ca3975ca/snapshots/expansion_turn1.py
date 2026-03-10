# Define a secure, ideally server-generated and managed, CSRF token value.
# For demonstration purposes, this is a constant. In a real application,
# this value should be unique per session or request and securely managed.
SECURE_CSRF_TOKEN_VALUE = "super_secret_and_unique_csrf_token"

def validate_csrf_protection(request_headers: dict) -> bool:
    """
    Validates the CSRF token from request headers.

    Args:
        request_headers: A dictionary representing the request headers.
                         Expected to contain 'X-CSRF-Token'.

    Returns:
        True if the CSRF token in the headers is valid, False otherwise.
    """
    token_from_header = request_headers.get("X-CSRF-Token")
    if token_from_header and token_from_header == SECURE_CSRF_TOKEN_VALUE:
        return True
    return False

if __name__ == '__main__':
    # Example Usage
    headers_with_valid_token = {
        "Content-Type": "application/json",
        "X-CSRF-Token": "super_secret_and_unique_csrf_token"
    }
    headers_with_invalid_token = {
        "Content-Type": "application/json",
        "X-CSRF-Token": "invalid_token_attempt"
    }
    headers_without_token = {
        "Content-Type": "application/json"
    }

    print(f"Validation with valid token: {validate_csrf_protection(headers_with_valid_token)}")
    print(f"Validation with invalid token: {validate_csrf_protection(headers_with_invalid_token)}")
    print(f"Validation without token: {validate_csrf_protection(headers_without_token)}")
