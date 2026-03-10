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
    # Check if project exists
    if proj_id not in PROJECTS_DB:
        raise PermissionError(f"Project {proj_id} not found")
    
    project = PROJECTS_DB[proj_id]
    
    # Check if member is part of the project
    if member_id not in project["members"]:
        raise PermissionError(f"User {member_id} is not a member of project {proj_id}")
    
    # Check role-based permissions
    if member_role == "viewer":
        raise PermissionError("Viewers do not have permission to modify project settings")
    
    if member_role == "team_member":
        # Team members can only modify certain settings
        restricted_settings = ["visibility", "project_name", "owner"]
        for setting in fresh_settings:
            if setting in restricted_settings:
                raise PermissionError(f"Team members cannot modify {setting}")
    
    # Update settings if all checks pass
    project["settings"].update(fresh_settings)
    
    return f"Project settings for {proj_id} have been successfully updated"
