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
    # Check if project exists
    if proj_id not in PROJECTS_DB:
        raise PermissionError(f"Project {proj_id} not found")
    
    project = PROJECTS_DB[proj_id]
    
    # Check if user is a member of the project
    if owner_id not in project["members"]:
        raise PermissionError(f"User {owner_id} is not a member of project {proj_id}")
    
    # Check access level permissions
    if access_level == "viewer":
        raise PermissionError("Viewers do not have permission to modify project settings")
    elif access_level == "team_member":
        # Team members can only modify if they are the creator
        if owner_id != project["creator_id"]:
            raise PermissionError("Team members can only modify projects they created")
    elif access_level == "project_manager":
        # Project managers can modify any project they're a member of
        pass
    else:
        raise PermissionError(f"Invalid access level: {access_level}")
    
    # Update the project settings
    project["settings"].update(modifications)
    
    return f"Project {proj_id} settings have been successfully updated"
