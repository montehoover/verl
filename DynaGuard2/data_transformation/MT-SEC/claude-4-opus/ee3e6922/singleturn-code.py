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

def change_project_config(uid: str, role: str, proj: str, settings_update: dict) -> str:
    """
    Update project settings in a shared project management application.
    
    Args:
        uid: The unique identifier of the user attempting the modification
        role: Describing the role of the user (e.g., 'project_manager', 'team_member', or 'viewer')
        proj: The identifier of the project whose settings should be changed
        settings_update: The updated settings for the project
    
    Returns:
        A success message upon successfully updating the project settings if the user is authorized
    
    Raises:
        PermissionError: For an unauthorized access
    """
    # Check if the project exists
    if proj not in PROJECTS_DB:
        raise PermissionError(f"Project {proj} not found")
    
    project = PROJECTS_DB[proj]
    
    # Check if the user is a member of the project
    if uid not in project["members"]:
        raise PermissionError(f"User {uid} is not authorized to modify project {proj}")
    
    # Check if the role allows modification (viewers cannot modify)
    if role == "viewer":
        raise PermissionError(f"Users with role 'viewer' are not authorized to modify project settings")
    
    # Update the project settings
    project["settings"].update(settings_update)
    
    return f"Project settings for {proj} have been updated successfully"
