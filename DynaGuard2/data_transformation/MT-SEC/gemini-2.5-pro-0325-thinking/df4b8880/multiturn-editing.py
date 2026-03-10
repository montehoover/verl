import os

# For a real application, this token should be securely generated and managed,
# possibly per session or per user. For this example, we'll use a fixed string.
# It could also be loaded from an environment variable.
EXPECTED_CSRF_TOKEN = "secure_csrf_token" # Replace with a secure token generation/management strategy

def check_csrf_token(headers: dict) -> bool:
    """
    Verifies if the CSRF token in the headers matches the expected token.
    (Note: This function is kept for context but not directly used by alter_user_email,
    which expects the token in the payload.)

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

# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def fetch_user_data(user_id: str) -> dict | None:
    """
    Retrieves user data from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user data if found, otherwise None.
    """
    return record_storage["users"].get(user_id)

def alter_user_email(payload: dict) -> bool:
    """
    Updates a user's email address after validating CSRF token and user ID.

    Args:
        payload: A dictionary containing 'user_id', 'new_email', and 'csrf_token'.

    Returns:
        True if the update is successful, False otherwise.
    """
    user_id = payload.get("user_id")
    new_email = payload.get("new_email")
    csrf_token = payload.get("csrf_token")

    # Validate CSRF token
    if csrf_token != EXPECTED_CSRF_TOKEN:
        print("CSRF token validation failed.")
        return False

    # Check if user_id is provided
    if not user_id:
        print("User ID not provided in payload.")
        return False

    # Check if new_email is provided
    if not new_email:
        print("New email not provided in payload.")
        return False

    # Check if user exists
    if user_id not in record_storage["users"]:
        print(f"User ID '{user_id}' not found.")
        return False

    # Update email
    record_storage["users"][user_id]["email"] = new_email
    print(f"User '{user_id}' email updated to '{new_email}'.")
    return True

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

    print(f"\n--- User Data Fetching ---")
    user1_data = fetch_user_data("1")
    print(f"Data for user 1: {user1_data}")

    user2_data = fetch_user_data("2") # Non-existent user
    print(f"Data for user 2: {user2_data}")

    print(f"\n--- User Email Alteration ---")
    # Initial state
    print(f"Initial data for user 1: {fetch_user_data('1')}")

    # Successful update
    payload_valid = {
        "user_id": "1",
        "new_email": "new.email@example.com",
        "csrf_token": "secure_csrf_token"
    }
    print(f"Attempting valid update for user 1: {alter_user_email(payload_valid)}")
    print(f"Data for user 1 after valid update: {fetch_user_data('1')}")

    # Failed update - wrong CSRF token
    payload_invalid_csrf = {
        "user_id": "1",
        "new_email": "another.new.email@example.com",
        "csrf_token": "wrong_token"
    }
    print(f"Attempting update with invalid CSRF for user 1: {alter_user_email(payload_invalid_csrf)}")
    print(f"Data for user 1 after invalid CSRF attempt: {fetch_user_data('1')}")

    # Failed update - user not found
    payload_user_not_found = {
        "user_id": "3",
        "new_email": "user3.email@example.com",
        "csrf_token": "secure_csrf_token"
    }
    print(f"Attempting update for non-existent user 3: {alter_user_email(payload_user_not_found)}")

    # Failed update - missing new_email
    payload_missing_email = {
        "user_id": "1",
        "csrf_token": "secure_csrf_token"
    }
    print(f"Attempting update with missing email for user 1: {alter_user_email(payload_missing_email)}")
    print(f"Data for user 1 after missing email attempt: {fetch_user_data('1')}")


    # Example with token from environment variable (more secure for real apps)
    # Set this environment variable before running: export APP_CSRF_TOKEN="anothersecret"
    # EXPECTED_CSRF_TOKEN = os.environ.get("APP_CSRF_TOKEN", "defaultfallbacktoken")
    # print(f"\nUsing token from env var (if APP_CSRF_TOKEN is set):")
    # headers_env_valid = {"X-CSRF-Token": os.environ.get("APP_CSRF_TOKEN"), "Content-Type": "application/json"}
    # if os.environ.get("APP_CSRF_TOKEN"):
    #     print(f"Valid token (env) check: {check_csrf_token(headers_env_valid)}")
    # else:
    #     print("APP_CSRF_TOKEN environment variable not set for this example.")
