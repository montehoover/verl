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
    # Check if project exists
    if pid not in PROJECTS_DB:
        raise PermissionError(f"Project {pid} not found")
    
    project = PROJECTS_DB[pid]
    
    # Check if user is authorized
    # Only project_manager role and creator can update settings
    if role != 'project_manager' and uid != project['creator_id']:
        raise PermissionError(f"User {uid} with role {role} is not authorized to update project {pid} settings")
    
    # Check if user is a member of the project
    if uid not in project['members']:
        raise PermissionError(f"User {uid} is not a member of project {pid}")
    
    # Update the settings
    project['settings'].update(settings)
    
    return f"Project {pid} settings successfully updated"
