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

def edit_project_settings(member_id: str, role: str, project_key: str, updated_configuration: dict) -> str:
    """
    Update project settings in a shared project management application.
    
    Args:
        member_id: The unique identifier of the user attempting the modification
        role: The role of the user (e.g., 'project_manager', 'team_member', or 'viewer')
        project_key: The identifier of the project whose settings should be changed
        updated_configuration: The updated settings for the project
    
    Returns:
        A success message upon successfully updating the project settings if the user is authorized
    
    Raises:
        PermissionError: For unauthorized access
    """
    # Check if project exists
    if project_key not in PROJECTS_DB:
        raise PermissionError(f"Project {project_key} not found")
    
    project = PROJECTS_DB[project_key]
    
    # Check if user is a member of the project
    if member_id not in project["members"]:
        raise PermissionError(f"User {member_id} is not a member of project {project_key}")
    
    # Check if user has the right role to edit settings
    if role != "project_manager":
        raise PermissionError(f"User with role '{role}' is not authorized to edit project settings")
    
    # Update the project settings
    project["settings"].update(updated_configuration)
    
    return f"Project settings for {project_key} have been updated successfully"
