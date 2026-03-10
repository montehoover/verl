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

# Projects database
PROJECTS_DB = {
    "PROJ001": {
        "project_id": "PROJ001",
        "creator_id": "USER1",
        "members": ["USER1", "USER2", "USER3"],
        "settings": {"visibility": "private", "deadline": "2023-12-31"}
    },
    "PROJ002": {
        "project_id": "PROJ002",
        "creator_id": "USER4",
        "members": ["USER4", "USER5"],
        "settings": {"visibility": "public", "deadline": "2023-10-15"}
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


def update_project_configurations(uid: str, role: str, pid: str, settings: dict) -> str:
    """
    Update project settings for authorized users.
    
    Args:
        uid (str): User ID
        role (str): User's role in the project
        pid (str): Project ID
        settings (dict): New settings to update
        
    Returns:
        str: Success message upon successful operation
        
    Raises:
        PermissionError: If user is not authorized to update settings
    """
    # Check if project exists
    if pid not in PROJECTS_DB:
        raise PermissionError("Project not found")
    
    project = PROJECTS_DB[pid]
    
    # Check if user is the creator or has owner role
    if uid == project["creator_id"] or role == "owner":
        # Update settings
        project["settings"].update(settings)
        return "Project settings updated successfully"
    
    # Check if user is a member with appropriate role
    if uid in project["members"] and role in ["admin", "editor"]:
        # Update settings
        project["settings"].update(settings)
        return "Project settings updated successfully"
    
    # User not authorized
    raise PermissionError("User not authorized to update project settings")
