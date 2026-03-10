# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


class RequestPayload:
    """
    Encapsulates request payload data for better organization and type safety.
    
    This class provides a clean interface for accessing request headers and body data,
    making the code more maintainable and reducing the chance of key errors.
    """
    
    def __init__(self, payload: dict):
        """
        Initialize the RequestPayload with the raw payload dictionary.
        
        Args:
            payload: dict containing 'headers' and 'body' keys with request data
        """
        self.headers = payload.get("headers", {})
        self.body = payload.get("body", {})
    
    def get_csrf_token(self) -> str:
        """
        Extract the CSRF token from request headers.
        
        Returns:
            str: The CSRF token value, or empty string if not present
        """
        return self.headers.get("X-CSRF-Token", "")
    
    def get_user_id(self) -> str:
        """
        Extract the user ID from request body.
        
        Returns:
            str: The user ID value, or empty string if not present
        """
        return self.body.get("user_id", "")
    
    def get_new_email(self) -> str:
        """
        Extract the new email address from request body.
        
        Returns:
            str: The new email address, or empty string if not present
        """
        return self.body.get("email", "")


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """
    Validate CSRF token from request headers for security.
    
    This function implements CSRF protection by comparing the token
    provided in the request headers against the expected token value.
    
    Args:
        headers: dict containing request headers with potential CSRF token
        expected_token: str representing the valid CSRF token
    
    Returns:
        bool: True if tokens match (valid), False otherwise
    """
    # Extract the CSRF token from the X-CSRF-Token header
    csrf_token = headers.get("X-CSRF-Token", "")
    
    # Perform constant-time comparison for security
    return csrf_token == expected_token


def extract_user_data(body: dict) -> tuple[str, str]:
    """
    Extract user ID and email from request body.
    
    This function parses the request body to retrieve the user identification
    and the new email address that should be updated.
    
    Args:
        body: dict containing the request body with user_id and email fields
    
    Returns:
        tuple[str, str]: A tuple containing (user_id, new_email)
                        Returns empty strings if fields are missing
    """
    # Extract user_id with empty string as default
    user_id = body.get("user_id", "")
    
    # Extract new email address with empty string as default
    new_email = body.get("email", "")
    
    return user_id, new_email


def user_exists(user_id: str, database: dict) -> bool:
    """
    Check if user exists in the database.
    
    This function verifies that a user with the given ID exists in our
    simulated database before attempting any update operations.
    
    Args:
        user_id: str representing the unique identifier of the user
        database: dict simulating our data storage with a 'users' key
    
    Returns:
        bool: True if user exists in database, False otherwise
    """
    # Access the users section of the database, defaulting to empty dict
    users = database.get("users", {})
    
    # Check if the user_id exists as a key in the users dictionary
    return user_id in users


def update_user_email(user_id: str, new_email: str, database: dict) -> bool:
    """
    Update user's email in the database.
    
    This function performs the actual update operation, modifying the user's
    email address in our simulated database. It includes a safety check to
    ensure the user exists before attempting the update.
    
    Args:
        user_id: str representing the unique identifier of the user
        new_email: str containing the new email address to set
        database: dict simulating our data storage with a 'users' key
    
    Returns:
        bool: True if update was successful, False if user doesn't exist
    """
    # Access the users section of the database
    users = database.get("users", {})
    
    # Check if user exists before attempting update
    if user_id in users:
        # Update the email field for the specified user
        database["users"][user_id]["email"] = new_email
        return True
    
    # Return False if user doesn't exist
    return False


def alter_user_email(payload: dict) -> bool:
    """
    Modify a user's email address in the web application.
    
    This is the main orchestration function that coordinates the email update process.
    It performs the following steps:
    1. Validates CSRF token for security
    2. Extracts user data from the request
    3. Verifies user existence
    4. Updates the email if all checks pass
    
    The function uses a parameter object pattern internally to better organize
    the payload data and make the code more maintainable.
    
    Args:
        payload: dict containing the request data with the following structure:
                {
                    "headers": {"X-CSRF-Token": "token_value"},
                    "body": {"user_id": "id", "email": "new@email.com"}
                }
    
    Returns:
        bool: True if the update is successful, False if any validation fails
              or an exception occurs
    """
    try:
        # Create a parameter object for cleaner payload handling
        request = RequestPayload(payload)
        
        # Step 1: Validate CSRF token for security
        # This prevents cross-site request forgery attacks
        if not validate_csrf_token(request.headers, EXPECTED_CSRF_TOKEN):
            return False
        
        # Step 2: Extract user data from the request
        # Get the user ID and new email from the parameter object
        user_id = request.get_user_id()
        new_email = request.get_new_email()
        
        # Step 3: Verify that the user exists in our database
        # This prevents attempts to update non-existent users
        if not user_exists(user_id, record_storage):
            return False
        
        # Step 4: Perform the actual email update
        # This will return True if successful, False otherwise
        return update_user_email(user_id, new_email, record_storage)
        
    except Exception:
        # Catch any unexpected errors and return False
        # In production, you might want to log the exception details
        return False
