def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user based on their user_id and password.
    
    Args:
        user_id (str): The user's ID
        password (str): The user's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # This is a basic implementation. In production, you would:
    # - Hash passwords using bcrypt or similar
    # - Store credentials in a secure database
    # - Use environment variables for sensitive data
    
    # Mock user database (for demonstration purposes)
    users = {
        "user123": "password123",
        "admin": "admin_pass",
        "john_doe": "secure_password"
    }
    
    # Check if user exists and password matches
    if user_id in users and users[user_id] == password:
        return True
    
    return False


# Mock projects database
PROJECTS_DB = {
    "proj001": {
        "name": "Website Redesign",
        "status": "active",
        "visibility": "private",
        "team_size": 5,
        "deadline": "2024-06-30",
        "budget": 50000
    },
    "proj002": {
        "name": "Mobile App Development",
        "status": "planning",
        "visibility": "public",
        "team_size": 8,
        "deadline": "2024-12-15",
        "budget": 120000
    },
    "proj003": {
        "name": "Data Migration",
        "status": "completed",
        "visibility": "private",
        "team_size": 3,
        "deadline": "2024-03-01",
        "budget": 30000
    }
}


def get_project_settings(proj_id: str) -> dict:
    """
    Retrieve the current settings of a project.
    
    Args:
        proj_id (str): The project ID
        
    Returns:
        dict: Project settings if found, empty dict otherwise
    """
    if proj_id in PROJECTS_DB:
        return PROJECTS_DB[proj_id].copy()
    
    return {}
