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
        bool: True if credentials are valid, False otherwise
    """
    # This is a simple example - in production, you would:
    # 1. Hash passwords using bcrypt or similar
    # 2. Store credentials in a secure database
    # 3. Use proper session management
    
    # Example hardcoded users for demonstration
    # In production, fetch from database and compare hashed passwords
    valid_users = {
        "user123": "password123",
        "admin": "adminpass",
        "john_doe": "securepass"
    }
    
    # Check if user exists and password matches
    if user_id in valid_users and valid_users[user_id] == password:
        return True
    
    return False


def validate_project_access(user_id: str, proj_id: str, access_level: str) -> bool:
    """
    Validate if a user has the required access level for a project.
    
    Args:
        user_id (str): The user's ID
        proj_id (str): The project's ID
        access_level (str): The required access level (e.g., 'read', 'write', 'admin')
        
    Returns:
        bool: True if user has required access level, False otherwise
    """
    # Example project access data structure
    # In production, this would be fetched from a database
    project_access = {
        "proj_001": {
            "user123": "admin",
            "john_doe": "write",
            "admin": "admin"
        },
        "proj_002": {
            "user123": "read",
            "john_doe": "admin",
            "admin": "admin"
        },
        "proj_003": {
            "admin": "admin"
        }
    }
    
    # Access level hierarchy: admin > write > read
    access_hierarchy = {
        "read": 1,
        "write": 2,
        "admin": 3
    }
    
    # Check if project exists
    if proj_id not in project_access:
        return False
    
    # Check if user has access to the project
    if user_id not in project_access[proj_id]:
        return False
    
    # Get user's actual access level for this project
    user_access = project_access[proj_id][user_id]
    
    # Compare access levels
    required_level = access_hierarchy.get(access_level, 0)
    user_level = access_hierarchy.get(user_access, 0)
    
    return user_level >= required_level


def change_project_settings(owner_id: str, access_level: str, proj_id: str, modifications: dict) -> str:
    """
    Change project settings for authorized users.
    
    Args:
        owner_id (str): The user's ID attempting the change
        access_level (str): The user's access level
        proj_id (str): The project's ID
        modifications (dict): Dictionary of settings to modify
        
    Returns:
        str: Success message
        
    Raises:
        PermissionError: If user lacks authorization
    """
    # Check if project exists
    if proj_id not in PROJECTS_DB:
        raise PermissionError(f"Project {proj_id} not found")
    
    project = PROJECTS_DB[proj_id]
    
    # Check if user is a member of the project
    if owner_id not in project["members"]:
        raise PermissionError(f"User {owner_id} is not a member of project {proj_id}")
    
    # Check access level - only 'admin' or project creator can modify settings
    if access_level != "admin" and owner_id != project["creator_id"]:
        raise PermissionError(f"User {owner_id} does not have permission to modify project {proj_id} settings")
    
    # Apply modifications
    for key, value in modifications.items():
        project["settings"][key] = value
    
    return f"Project {proj_id} settings successfully updated"
