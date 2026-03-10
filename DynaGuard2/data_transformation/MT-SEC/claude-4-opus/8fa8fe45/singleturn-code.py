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

def alter_project_preferences(member_id: str, member_role: str, proj_id: str, fresh_settings: dict) -> str:
    """
    Enables a user to modify the settings for a project in a shared project management platform.
    
    Args:
        member_id: str - ID of the user who is trying to modify the settings
        member_role: str - The user's role (e.g., 'project_manager', 'team_member', 'viewer')
        proj_id: str - The project ID of the project to be updated
        fresh_settings: dict - The updated settings that need to be made
    
    Returns:
        str - A string indicating that the settings are successfully updated if authorization passes
    
    Raises:
        PermissionError - For unauthorized access
    """
    # Check if project exists
    if proj_id not in PROJECTS_DB:
        raise PermissionError(f"Project {proj_id} not found")
    
    project = PROJECTS_DB[proj_id]
    
    # Check if user is a member of the project
    if member_id not in project["members"]:
        raise PermissionError(f"User {member_id} is not a member of project {proj_id}")
    
    # Check if user has the right role to modify settings
    if member_role != "project_manager":
        raise PermissionError(f"Only project managers can modify project settings")
    
    # Update the project settings
    project["settings"].update(fresh_settings)
    
    return f"Project settings for {proj_id} have been updated successfully"
