def authenticate_user(member_id: str, password: str) -> bool:
    """
    Authenticate a user based on member_id and password.
    
    Args:
        member_id (str): The member's unique identifier
        password (str): The member's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # Temporary hardcoded credentials for demonstration
    # In production, this should check against a secure database
    # with properly hashed passwords
    valid_credentials = {
        "member001": "securepass123",
        "member002": "projectpass456",
        "admin": "adminpass789"
    }
    
    # Check if member_id exists and password matches
    if member_id in valid_credentials:
        return valid_credentials[member_id] == password
    
    return False


def get_project_settings(project_key: str, member_id: str) -> dict | None:
    """
    Retrieve project settings if the member is part of the project.
    
    Args:
        project_key (str): The project's unique identifier
        member_id (str): The member's unique identifier
        
    Returns:
        dict | None: Project settings if member has access, None otherwise
    """
    # Mock project membership data
    project_members = {
        "proj_alpha": ["member001", "member002", "admin"],
        "proj_beta": ["member002", "admin"],
        "proj_gamma": ["admin"]
    }
    
    # Mock project settings data
    project_settings = {
        "proj_alpha": {
            "name": "Project Alpha",
            "description": "Main development project",
            "status": "active",
            "visibility": "private",
            "created_date": "2024-01-15",
            "deadline": "2024-12-31"
        },
        "proj_beta": {
            "name": "Project Beta",
            "description": "Research initiative",
            "status": "planning",
            "visibility": "internal",
            "created_date": "2024-02-01",
            "deadline": "2024-09-30"
        },
        "proj_gamma": {
            "name": "Project Gamma",
            "description": "Infrastructure upgrade",
            "status": "active",
            "visibility": "confidential",
            "created_date": "2024-03-01",
            "deadline": "2024-06-30"
        }
    }
    
    # Check if project exists
    if project_key not in project_members:
        return None
    
    # Check if member is part of the project
    if member_id not in project_members[project_key]:
        return None
    
    # Return project settings if member has access
    return project_settings.get(project_key)
