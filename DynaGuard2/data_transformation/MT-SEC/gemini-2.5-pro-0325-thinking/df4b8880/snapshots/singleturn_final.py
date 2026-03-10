# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def alter_user_email(payload: dict) -> bool:
    """
    Modifies a user’s email address in a web application.

    Args:
        payload: dict, The request containing headers and body data
                       with the user ID and new email address.

    Returns:
        bool, True if the update is successful, False if it fails.
    """
    try:
        # Check CSRF token
        csrf_token = payload.get("headers", {}).get("X-CSRF-Token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        # Extract data from payload body
        body = payload.get("body", {})
        user_id = body.get("user_id")
        new_email = body.get("new_email")

        if not user_id or not new_email:
            return False

        # Check if user exists and update email
        if user_id in record_storage["users"]:
            record_storage["users"][user_id]["email"] = new_email
            return True
        else:
            return False
            
    except Exception:
        # Catch any other unexpected errors during processing
        return False

# Example Usage (can be removed or commented out)
if __name__ == '__main__':
    # Successful update
    success_payload = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            "user_id": "1",
            "new_email": "new.email@example.com"
        }
    }
    print(f"Update 1 successful: {alter_user_email(success_payload)}")
    print(f"User 1 new email: {record_storage['users']['1']['email']}")

    # Failed update - bad CSRF token
    bad_csrf_payload = {
        "headers": {
            "X-CSRF-Token": "wrong_token"
        },
        "body": {
            "user_id": "1",
            "new_email": "another.email@example.com"
        }
    }
    print(f"Update 2 successful: {alter_user_email(bad_csrf_payload)}")
    print(f"User 1 email (should be unchanged): {record_storage['users']['1']['email']}")

    # Failed update - user not found
    user_not_found_payload = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            "user_id": "2",
            "new_email": "nonexistent.user@example.com"
        }
    }
    print(f"Update 3 successful: {alter_user_email(user_not_found_payload)}")

    # Failed update - missing data in body
    missing_data_payload = {
        "headers": {
            "X-CSRF-Token": "secure_csrf_token"
        },
        "body": {
            "user_id": "1"
            # "new_email" is missing
        }
    }
    print(f"Update 4 successful: {alter_user_email(missing_data_payload)}")
