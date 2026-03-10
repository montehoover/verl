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

ACCESS_LOG = []


def log_project_access(uid: str, role: str, proj: str):
    # Define allowed roles
    allowed_roles = {'project_manager', 'team_member'}
    
    # Check if role allows viewing
    can_view = role in allowed_roles
    
    # Log the access attempt
    log_entry = {
        'user_id': uid,
        'role': role,
        'project_id': proj,
        'access_granted': can_view,
        'action': 'view_settings'
    }
    ACCESS_LOG.append(log_entry)
    
    # Raise error if unauthorized
    if not can_view:
        raise PermissionError(f"User {uid} with role '{role}' is not authorized to view project settings")
    
    # Return project settings if authorized
    if proj in PROJECTS_DB:
        return PROJECTS_DB[proj]['settings']
    else:
        return None
