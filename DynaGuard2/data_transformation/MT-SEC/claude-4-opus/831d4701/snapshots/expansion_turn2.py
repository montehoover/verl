import hashlib

# Mock user database (in production, this would be a real database)
USER_DATABASE = {
    "user123": "5f4dcc3b5aa765d61d8327deb882cf99",  # password: "password"
    "admin": "21232f297a57a5a743894a0e4a801fc3",     # password: "admin"
    "john_doe": "5d41402abc4b2a76b9719d911017c592"   # password: "hello"
}

# Mock project membership database
PROJECT_MEMBERSHIPS = {
    "project_001": {
        "user123": "member",
        "admin": "owner",
        "john_doe": "viewer"
    },
    "project_002": {
        "user123": "owner",
        "john_doe": "member"
    },
    "project_003": {
        "admin": "owner"
    }
}

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user based on user_id and password.
    
    Args:
        user_id (str): The user's ID
        password (str): The user's password
        
    Returns:
        bool: True if credentials are correct, False otherwise
    """
    if user_id not in USER_DATABASE:
        return False
    
    # Hash the provided password
    password_hash = hashlib.md5(password.encode()).hexdigest()
    
    # Compare with stored hash
    return USER_DATABASE[user_id] == password_hash


def verify_project_membership(user_id: str, project_id: str, role: str) -> bool:
    """
    Verify if a user has a specific role within a project.
    
    Args:
        user_id (str): The user's ID
        project_id (str): The project's ID
        role (str): The role to verify (e.g., 'owner', 'member', 'viewer')
        
    Returns:
        bool: True if the user has the specified role in the project, False otherwise
    """
    if project_id not in PROJECT_MEMBERSHIPS:
        return False
    
    if user_id not in PROJECT_MEMBERSHIPS[project_id]:
        return False
    
    return PROJECT_MEMBERSHIPS[project_id][user_id] == role
