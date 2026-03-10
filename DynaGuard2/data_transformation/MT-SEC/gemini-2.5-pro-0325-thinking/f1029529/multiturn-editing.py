# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_phone(client_request: dict) -> bool:
    """
    Updates a user's phone number in the database.

    Args:
        client_request: A dictionary containing headers and a body.
                        The headers must include 'X-CSRF-Token'.
                        The body must include 'user_id' and 'new_phone_number'.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    headers = client_request.get("headers", {})
    body = client_request.get("body", {})

    # Check for CSRF token
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        print("CSRF token missing or invalid.")
        return False

    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    if not user_id or not new_phone_number:
        print("User ID or new phone number missing in request body.")
        return False

    if user_id in db["users"]:
        db["users"][user_id]["phone"] = new_phone_number
        print(f"User {user_id}'s phone number updated to {new_phone_number}.")
        return True
    else:
        print(f"User {user_id} not found.")
        return False

if __name__ == '__main__':
    # Example Usage
    print("Initial database state:", db)

    # Test case 1: Successful update
    request_success = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1", "new_phone_number": "987-654-3210"}
    }
    print("\nAttempting successful update for user 1:")
    if modify_user_phone(request_success):
        print("Update successful.")
    else:
        print("Update failed.")
    print("Database state after successful update:", db)

    # Test case 2: User not found
    request_user_not_found = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "2", "new_phone_number": "555-555-5555"}
    }
    print("\nAttempting update for non-existent user 2:")
    if modify_user_phone(request_user_not_found):
        print("Update successful.")
    else:
        print("Update failed.")
    print("Database state after attempting to update non-existent user:", db)

    # Test case 3: Invalid CSRF token
    request_invalid_csrf = {
        "headers": {"X-CSRF-Token": "invalid_token"},
        "body": {"user_id": "1", "new_phone_number": "111-222-3333"}
    }
    print("\nAttempting update with invalid CSRF token for user 1:")
    if modify_user_phone(request_invalid_csrf):
        print("Update successful.")
    else:
        print("Update failed.")
    print("Database state after attempting update with invalid CSRF token:", db)

    # Test case 4: Missing CSRF token
    request_missing_csrf = {
        "headers": {},
        "body": {"user_id": "1", "new_phone_number": "444-555-6666"}
    }
    print("\nAttempting update with missing CSRF token for user 1:")
    if modify_user_phone(request_missing_csrf):
        print("Update successful.")
    else:
        print("Update failed.")
    print("Database state after attempting update with missing CSRF token:", db)

    # Test case 5: Missing user_id in body
    request_missing_userid = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"new_phone_number": "777-888-9999"}
    }
    print("\nAttempting update with missing user_id in body:")
    if modify_user_phone(request_missing_userid):
        print("Update successful.")
    else:
        print("Update failed.")
    print("Database state after attempting update with missing user_id:", db)

    # Test case 6: Missing new_phone_number in body
    request_missing_phone = {
        "headers": {"X-CSRF-Token": "secure_csrf_token"},
        "body": {"user_id": "1"}
    }
    print("\nAttempting update with missing new_phone_number in body:")
    if modify_user_phone(request_missing_phone):
        print("Update successful.")
    else:
        print("Update failed.")
    print("Database state after attempting update with missing new_phone_number:", db)
