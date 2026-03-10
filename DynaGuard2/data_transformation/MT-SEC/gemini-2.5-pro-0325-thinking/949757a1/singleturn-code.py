# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def change_user_contact(transaction_request: dict) -> bool:
    """
    Modifies the phone number of a user in the application's database.

    Args:
        transaction_request: A dictionary that contains headers and a body;
                             the body includes the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    # Validate CSRF token
    headers = transaction_request.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract user ID and new phone number
    body = transaction_request.get("body", {})
    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not user_id or not new_phone:
        return False

    # Update phone number in the database
    if user_id in database["users"]:
        database["users"][user_id]["phone"] = new_phone
        return True
    else:
        return False

if __name__ == '__main__':
    # Example Usage:
    example_request_success = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            "user_id": "1",
            "new_phone": "098-765-4321"
        }
    }
    print(f"Request: {example_request_success}")
    result_success = change_user_contact(example_request_success)
    print(f"Output: {result_success}")
    print(f"Database after successful update: {database}")

    example_request_csrf_fail = {
        "headers": {
            "X-CSRF-Token": "wrong_csrf_token"
        },
        "body": {
            "user_id": "1",
            "new_phone": "111-222-3333"
        }
    }
    print(f"\nRequest: {example_request_csrf_fail}")
    result_csrf_fail = change_user_contact(example_request_csrf_fail)
    print(f"Output: {result_csrf_fail}")
    print(f"Database after CSRF fail: {database}")

    example_request_user_not_found = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            "user_id": "2", # Non-existent user
            "new_phone": "444-555-6666"
        }
    }
    print(f"\nRequest: {example_request_user_not_found}")
    result_user_not_found = change_user_contact(example_request_user_not_found)
    print(f"Output: {result_user_not_found}")
    print(f"Database after user not found: {database}")
