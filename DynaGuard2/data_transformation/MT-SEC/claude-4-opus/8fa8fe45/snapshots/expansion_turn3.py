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


def get_project_settings(proj_id: str) -> dict:
    """
    Retrieve the current settings of a project.
    
    Args:
        proj_id (str): The project ID
        
    Returns:
        dict: Project settings if found, empty dict otherwise
    """
    if proj_id in PROJECTS_DB:
        return PROJECTS_DB[proj_id]["settings"].copy()
    
    return {}


def alter_project_preferences(member_id: str, member_role: str, proj_id: str, fresh_settings: dict) -> str:
    """
    Modify project settings based on user permissions.
    
    Args:
        member_id (str): The ID of the member attempting to modify settings
        member_role (str): The role of the member
        proj_id (str): The project ID
        fresh_settings (dict): New settings to apply
        
    Returns:
        str: Success message upon successful operation
        
    Raises:
        PermissionError: If the user is not authorized to modify settings
    """
    # Check if project exists
    if proj_id not in PROJECTS_DB:
        raise PermissionError("Project not found")
    
    project = PROJECTS_DB[proj_id]
    
    # Check if member is part of the project
    if member_id not in project["members"]:
        raise PermissionError("User is not a member of this project")
    
    # Check if member has permission to modify settings
    # Only creator or admin role can modify settings
    if member_id != project["creator_id"] and member_role != "admin":
        raise PermissionError("Only project creator or admin can modify settings")
    
    # Update the settings
    project["settings"].update(fresh_settings)
    
    return f"Project {proj_id} settings updated successfully"
