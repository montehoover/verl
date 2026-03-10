# Example PROJECTS_DB that would be available in the actual setup
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

def edit_project_settings(member_id, role, project_key, updated_configuration):
    """
    Allows users to update project settings.
    
    Args:
        member_id (str): The unique identifier of the user attempting the modification
        role (str): The user's role
        project_key (str): The identifier of the project
        updated_configuration (dict): The new settings for the project
        
    Returns:
        str: Success message if the update is successful
        
    Raises:
        PermissionError: If the user doesn't have permission to edit settings
    """
    # Check if project exists
    if project_key not in PROJECTS_DB:
        raise PermissionError(f"Project {project_key} not found")
    
    project = PROJECTS_DB[project_key]
    
    # Check if user is a member of the project
    if member_id not in project["members"]:
        raise PermissionError(f"User {member_id} is not a member of project {project_key}")
    
    # Check if user has permission (only creator can edit settings)
    if member_id != project["creator_id"]:
        raise PermissionError(f"User {member_id} does not have permission to edit settings for project {project_key}")
    
    # Update the settings
    project["settings"].update(updated_configuration)
    
    return f"Settings for project {project_key} updated successfully"
