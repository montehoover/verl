# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def get_user_details(user_request):
    """
    Retrieves a user's email address from the database.

    Args:
        user_request: A dictionary containing a body with the user ID.
                      Example: {"body": {"user_id": "1"}}

    Returns:
        The email address (str) for the given user ID, or None if not found.
    """
    try:
        user_id = user_request['body']['user_id']
        user_info = database.get("users", {}).get(user_id)
        if user_info:
            return user_info.get("email")
        return None
    except (KeyError, TypeError):
        # Handle cases where 'body' or 'user_id' might be missing, or user_request is not a dict
        return None

if __name__ == '__main__':
    # Example Usage
    request1 = {"body": {"user_id": "1"}}
    email1 = get_user_details(request1)
    print(f"User ID 1 email: {email1}") # Expected: user1@example.com

    request2 = {"body": {"user_id": "5"}}
    email2 = get_user_details(request2)
    print(f"User ID 5 email: {email2}") # Expected: None

    request3 = {"body": {}} # Missing user_id
    email3 = get_user_details(request3)
    print(f"Invalid request (missing user_id) email: {email3}") # Expected: None

    request4 = {} # Missing body
    email4 = get_user_details(request4)
    print(f"Invalid request (missing body) email: {email4}") # Expected: None

    request5 = "not a dict"
    email5 = get_user_details(request5)
    print(f"Invalid request (not a dict) email: {email5}") # Expected: None
