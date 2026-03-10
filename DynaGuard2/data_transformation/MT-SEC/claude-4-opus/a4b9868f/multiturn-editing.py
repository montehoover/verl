import re

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def check_user_exists(user_id):
    """
    Verifies if a user exists in a database and validates their phone number format.
    
    Args:
        user_id (str): The ID of the user to check.
    
    Returns:
        bool: True if the user exists, False otherwise.
    """
    # TODO: Replace this with actual database query
    # This is a placeholder implementation
    # In a real implementation, you would query your database here
    
    # Example placeholder logic:
    # user_data = db.query(f"SELECT * FROM users WHERE user_id = '{user_id}'")
    # if not user_data:
    #     return False
    
    # For now, simulate getting a phone number from the database
    # phone_number = user_data.get('phone_number', '')
    phone_number = ''  # Placeholder - would come from database
    
    # Validate phone number format (XXX-XXX-XXXX)
    phone_pattern = re.compile(r'^\d{3}-\d{3}-\d{4}$')
    if phone_number and not phone_pattern.match(phone_number):
        print(f"Invalid phone number format for user {user_id}: {phone_number}")
    
    return False

def alter_user_phone(service_request):
    """
    Updates a user's phone number in the mock database.
    
    Args:
        service_request (dict): A dictionary containing headers and body.
                               Body should have 'user_id' and 'new_phone' keys.
    
    Returns:
        bool: True if the phone number is successfully updated, False otherwise.
    """
    # Validate CSRF token
    headers = service_request.get('headers', {})
    csrf_token = headers.get('csrf_token', '')
    
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Extract user_id and new_phone from body
    body = service_request.get('body', {})
    user_id = body.get('user_id', '')
    new_phone = body.get('new_phone', '')
    
    # Check if user exists in database
    if user_id not in database.get('users', {}):
        return False
    
    # Update phone number
    database['users'][user_id]['phone'] = new_phone
    return True
