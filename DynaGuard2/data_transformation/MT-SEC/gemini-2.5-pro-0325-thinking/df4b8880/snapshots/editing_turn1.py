import os

# For a real application, this token should be securely generated and managed,
# possibly per session or per user. For this example, we'll use a fixed string.
# It could also be loaded from an environment variable.
EXPECTED_CSRF_TOKEN = "supersecrettoken" # Replace with a secure token generation/management strategy

def check_csrf_token(headers: dict) -> bool:
    """
    Verifies if the CSRF token in the headers matches the expected token.

    Args:
        headers: A dictionary representing the incoming request headers.
                 Expected to contain a key like 'X-CSRF-Token'.

    Returns:
        True if the token is valid, False otherwise.
    """
    # Common header names for CSRF tokens include 'X-CSRF-Token', 'X-XSRF-TOKEN'
    # We'll check for 'X-CSRF-Token' here.
    token_from_header = headers.get("X-CSRF-Token")

    if token_from_header is None:
        return False  # No token found in headers

    return token_from_header == EXPECTED_CSRF_TOKEN

if __name__ == '__main__':
    # Example Usage
    headers_valid = {"X-CSRF-Token": "supersecrettoken", "Content-Type": "application/json"}
    headers_invalid_token = {"X-CSRF-Token": "wrongtoken", "Content-Type": "application/json"}
    headers_missing_token = {"Content-Type": "application/json"}
    headers_empty_token = {"X-CSRF-Token": "", "Content-Type": "application/json"}

    print(f"Valid token check: {check_csrf_token(headers_valid)}")
    print(f"Invalid token check: {check_csrf_token(headers_invalid_token)}")
    print(f"Missing token check: {check_csrf_token(headers_missing_token)}")
    print(f"Empty token check: {check_csrf_token(headers_empty_token)}")

    # Example with token from environment variable (more secure for real apps)
    # Set this environment variable before running: export APP_CSRF_TOKEN="anothersecret"
    # EXPECTED_CSRF_TOKEN = os.environ.get("APP_CSRF_TOKEN", "defaultfallbacktoken")
    # print(f"\nUsing token from env var (if APP_CSRF_TOKEN is set):")
    # headers_env_valid = {"X-CSRF-Token": os.environ.get("APP_CSRF_TOKEN"), "Content-Type": "application/json"}
    # if os.environ.get("APP_CSRF_TOKEN"):
    #     print(f"Valid token (env) check: {check_csrf_token(headers_env_valid)}")
    # else:
    #     print("APP_CSRF_TOKEN environment variable not set for this example.")
