# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def verify_user_exists(user_request):
    """
    Verifies if a user ID exists in the database.

    Args:
        user_request: A dictionary containing a body with the user ID.
                      Example: {"body": {"user_id": "1"}}

    Returns:
        True if the user ID exists, False otherwise.
    """
    try:
        user_id = user_request['body']['user_id']
        # Check if user_id is a key in the "users" dictionary
        if user_id in database.get("users", {}):
            return True
        return False
    except (KeyError, TypeError):
        # Handle cases where 'body' or 'user_id' might be missing, or user_request is not a dict
        return False

if __name__ == '__main__':
    # Example Usage
    request1 = {"body": {"user_id": "1"}}
    exists1 = verify_user_exists(request1)
    print(f"User ID 1 exists: {exists1}") # Expected: True

    request2 = {"body": {"user_id": "5"}}
    exists2 = verify_user_exists(request2)
    print(f"User ID 5 exists: {exists2}") # Expected: False

    request3 = {"body": {}} # Missing user_id
    exists3 = verify_user_exists(request3)
    print(f"Invalid request (missing user_id) exists: {exists3}") # Expected: False

    request4 = {} # Missing body
    exists4 = verify_user_exists(request4)
    print(f"Invalid request (missing body) exists: {exists4}") # Expected: False

    request5 = "not a dict"
    exists5 = verify_user_exists(request5)
    print(f"Invalid request (not a dict) exists: {exists5}") # Expected: False
