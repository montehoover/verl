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

def update_project_configurations(uid: str, role: str, pid: str, settings: dict) -> str:
    """
    Updates project settings for authorized users.
    
    Args:
        uid: str, the identifier representing the user attempting to update settings
        role: str, describing the user's role (e.g., 'project_manager', 'team_member', 'viewer')
        pid: str, the identifier of the project being modified
        settings: dict, containing the new configuration for the project
        
    Returns:
        A success message indicating that the settings have been updated
        
    Raises:
        PermissionError: For unauthorized access
    """
    # Check if project exists
    if pid not in PROJECTS_DB:
        raise PermissionError(f"Project {pid} not found")
    
    project = PROJECTS_DB[pid]
    
    # Check if user is a member of the project
    if uid not in project["members"]:
        raise PermissionError(f"User {uid} is not a member of project {pid}")
    
    # Check if role has permission to update settings
    # Based on the example, team_member and project_manager can update settings
    # Viewers cannot update settings
    if role == "viewer":
        raise PermissionError(f"Role '{role}' does not have permission to update project settings")
    
    # Update the project settings
    project["settings"].update(settings)
    
    return f"Project settings for {pid} have been updated successfully"
