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


def edit_project_settings(member_id: str, role: str, project_key: str, updated_configuration: dict) -> str:
    """
    Update project settings if the user has appropriate permissions.
    
    Args:
        member_id (str): The member's unique identifier
        role (str): The member's role in the project
        project_key (str): The project's unique identifier
        updated_configuration (dict): Dictionary containing settings to update
        
    Returns:
        str: Success message upon successful update
        
    Raises:
        PermissionError: If user is not authorized to edit settings
    """
    # Check if project exists
    if project_key not in PROJECTS_DB:
        raise PermissionError(f"Project {project_key} not found")
    
    project = PROJECTS_DB[project_key]
    
    # Check if member is part of the project
    if member_id not in project["members"]:
        raise PermissionError(f"User {member_id} is not a member of project {project_key}")
    
    # Check if member has permission to edit settings
    # Only creator or users with admin/manager roles can edit settings
    if member_id != project["creator_id"] and role not in ["admin", "manager"]:
        raise PermissionError(f"User {member_id} does not have permission to edit project settings")
    
    # Update the project settings
    project["settings"].update(updated_configuration)
    
    return f"Project settings for {project_key} successfully updated"
