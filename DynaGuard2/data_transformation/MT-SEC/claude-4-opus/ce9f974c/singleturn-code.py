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

def change_project_settings(owner_id: str, access_level: str, proj_id: str, modifications: dict) -> str:
    """
    Allows authorized users to update project settings within a project management platform.
    
    Args:
        owner_id: str - The identifier representing the user attempting to update settings
        access_level: str - Describing the user's role (e.g., 'project_manager', 'team_member', 'viewer')
        proj_id: str - The identifier of the project being modified
        modifications: dict - Containing the new configuration for the project
    
    Returns:
        str - A success message indicating that the settings have been updated
    
    Raises:
        PermissionError - For unauthorized access
    """
    # Check if project exists
    if proj_id not in PROJECTS_DB:
        raise PermissionError(f"Project {proj_id} not found")
    
    project = PROJECTS_DB[proj_id]
    
    # Check if user is a member of the project
    if owner_id not in project["members"]:
        raise PermissionError(f"User {owner_id} is not a member of project {proj_id}")
    
    # Check if user has sufficient access level
    # Only project_manager or the creator can modify settings
    is_creator = owner_id == project["creator_id"]
    is_project_manager = access_level == "project_manager"
    
    if not (is_creator or is_project_manager):
        raise PermissionError(f"User {owner_id} does not have permission to modify project settings")
    
    # Update the project settings
    project["settings"].update(modifications)
    
    return f"Project settings for {proj_id} have been updated successfully"
