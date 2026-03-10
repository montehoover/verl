# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def get_user_info(user_id):
    """
    Retrieves a user's information from a database using their user ID.
    
    Args:
        user_id (str): The unique identifier of the user
        
    Returns:
        dict: A dictionary containing the user's details if found, None otherwise
    """
    # This is a placeholder implementation
    # In a real application, this would connect to a database
    
    # Example mock database
    users_db = {
        "user123": {
            "id": "user123",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 30,
            "created_at": "2024-01-15"
        },
        "user456": {
            "id": "user456",
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "age": 28,
            "created_at": "2024-02-20"
        }
    }
    
    return users_db.get(user_id)


def update_user_email(user_id, new_email):
    """
    Updates a user's email address in the database.
    
    Args:
        user_id (str): The unique identifier of the user
        new_email (str): The new email address to set
        
    Returns:
        bool: True if the update is successful, False if the user is not found
    """
    # This is a placeholder implementation
    # In a real application, this would connect to a database
    
    # Example mock database
    users_db = {
        "user123": {
            "id": "user123",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 30,
            "created_at": "2024-01-15"
        },
        "user456": {
            "id": "user456",
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "age": 28,
            "created_at": "2024-02-20"
        }
    }
    
    if user_id in users_db:
        users_db[user_id]["email"] = new_email
        return True
    return False


def modify_user_phone(client_request):
    """
    Updates a user's phone number in the database after validating CSRF token.
    
    Args:
        client_request (dict): A dictionary containing headers and body with user ID and new phone number
        
    Returns:
        bool: True if the phone number is successfully updated, False otherwise
    """
    # Check if headers exist and contain CSRF token
    if "headers" not in client_request:
        return False
    
    headers = client_request["headers"]
    if "csrf_token" not in headers:
        return False
    
    # Validate CSRF token
    if headers["csrf_token"] != EXPECTED_CSRF_TOKEN:
        return False
    
    # Check if body exists and contains required fields
    if "body" not in client_request:
        return False
    
    body = client_request["body"]
    if "user_id" not in body or "new_phone" not in body:
        return False
    
    user_id = body["user_id"]
    new_phone = body["new_phone"]
    
    # Update phone number in database
    if user_id in db["users"]:
        db["users"][user_id]["phone"] = new_phone
        return True
    
    return False
