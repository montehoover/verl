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


def check_project_exists(proj_id: str, projects_db: dict) -> dict:
    """Check if project exists and return it."""
    if proj_id not in projects_db:
        raise PermissionError(f"Project {proj_id} not found")
    return projects_db[proj_id]


def check_member_authorization(member_id: str, member_role: str, project: dict, fresh_settings: dict) -> None:
    """Check if member is authorized to make changes."""
    # Check if member is part of the project
    if member_id not in project["members"]:
        raise PermissionError(f"User {member_id} is not a member of project {project['project_id']}")
    
    # Check role-based permissions
    if member_role == "viewer":
        raise PermissionError("Viewers do not have permission to modify project settings")
    
    if member_role == "team_member":
        # Team members can only modify certain settings
        restricted_settings = ["visibility", "project_name", "owner"]
        for setting in fresh_settings:
            if setting in restricted_settings:
                raise PermissionError(f"Team members cannot modify {setting}")


def update_project_settings(project: dict, fresh_settings: dict) -> None:
    """Update project settings."""
    project["settings"].update(fresh_settings)


def alter_project_preferences(member_id: str, member_role: str, proj_id: str, fresh_settings: dict) -> str:
    # Check if project exists
    project = check_project_exists(proj_id, PROJECTS_DB)
    
    # Check authorization
    check_member_authorization(member_id, member_role, project, fresh_settings)
    
    # Update settings
    update_project_settings(project, fresh_settings)
    
    return f"Project settings for {proj_id} have been successfully updated"
