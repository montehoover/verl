# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_phone(user_request: dict) -> bool:
    """
    Modifies the phone number of a user in the database.

    Args:
        user_request: A dictionary containing headers and a body.
                      The body includes the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    headers = user_request.get("headers", {})
    body = user_request.get("body", {})

    # Validate CSRF token
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not user_id or not new_phone:
        return False

    if user_id in database["users"]:
        database["users"][user_id]["phone"] = new_phone
        return True
    
    return False

# Example Usage (optional, for testing)
if __name__ == '__main__':
    # Test case 1: Successful update
    request1 = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            "user_id": "1",
            "new_phone": "098-765-4321"
        }
    }
    print(f"Test Case 1 Input: {request1}")
    result1 = modify_user_phone(request1)
    print(f"Test Case 1 Output: {result1}") # Expected: True
    print(f"Database after Test Case 1: {database}") # Expected: {'users': {'1': {'phone': '098-765-4321'}}}
    print("-" * 20)

    # Test case 2: Invalid CSRF token
    request2 = {
        "headers": {
            "X-CSRF-Token": "invalid_token"
        },
        "body": {
            "user_id": "1",
            "new_phone": "111-222-3333"
        }
    }
    print(f"Test Case 2 Input: {request2}")
    result2 = modify_user_phone(request2)
    print(f"Test Case 2 Output: {result2}") # Expected: False
    print(f"Database after Test Case 2: {database}") # Expected: (should be unchanged from previous successful update)
    print("-" * 20)

    # Test case 3: User ID not found
    request3 = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            "user_id": "2", # Non-existent user
            "new_phone": "444-555-6666"
        }
    }
    print(f"Test Case 3 Input: {request3}")
    result3 = modify_user_phone(request3)
    print(f"Test Case 3 Output: {result3}") # Expected: False
    print(f"Database after Test Case 3: {database}")
    print("-" * 20)

    # Test case 4: Missing user_id in body
    request4 = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            # "user_id": "1", # Missing
            "new_phone": "777-888-9999"
        }
    }
    print(f"Test Case 4 Input: {request4}")
    result4 = modify_user_phone(request4)
    print(f"Test Case 4 Output: {result4}") # Expected: False
    print(f"Database after Test Case 4: {database}")
    print("-" * 20)

    # Test case 5: Missing new_phone in body
    request5 = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            "user_id": "1",
            # "new_phone": "777-888-9999" # Missing
        }
    }
    print(f"Test Case 5 Input: {request5}")
    result5 = modify_user_phone(request5)
    print(f"Test Case 5 Output: {result5}") # Expected: False
    print(f"Database after Test Case 5: {database}")
    print("-" * 20)

    # Test case 6: Empty body
    request6 = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {}
    }
    print(f"Test Case 6 Input: {request6}")
    result6 = modify_user_phone(request6)
    print(f"Test Case 6 Output: {result6}") # Expected: False
    print(f"Database after Test Case 6: {database}")
    print("-" * 20)

    # Test case 7: Empty headers (implies missing CSRF)
    request7 = {
        "headers": {},
        "body": {
            "user_id": "1",
            "new_phone": "123-123-1234"
        }
    }
    print(f"Test Case 7 Input: {request7}")
    result7 = modify_user_phone(request7)
    print(f"Test Case 7 Output: {result7}") # Expected: False
    print(f"Database after Test Case 7: {database}")
