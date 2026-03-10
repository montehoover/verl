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

def check_authorization(member_id: str, role: str, project_key: str, project: dict) -> None:
    # Check if user is a member of the project
    if member_id not in project["members"]:
        raise PermissionError(f"User {member_id} is not a member of project {project_key}")
    
    # Check if user has permission based on role
    if role == "viewer":
        raise PermissionError(f"User with role '{role}' does not have permission to edit project settings")
    
    if role not in ["project_manager", "team_member"]:
        raise PermissionError(f"Invalid role '{role}'")

def update_settings(project: dict, updated_configuration: dict) -> None:
    project["settings"].update(updated_configuration)

def edit_project_settings(member_id: str, role: str, project_key: str, updated_configuration: dict) -> str:
    # Check if project exists
    if project_key not in PROJECTS_DB:
        raise PermissionError(f"Project {project_key} not found")
    
    project = PROJECTS_DB[project_key]
    
    check_authorization(member_id, role, project_key, project)
    update_settings(project, updated_configuration)
    
    return f"Successfully updated settings for project {project_key}"
