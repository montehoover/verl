import datetime

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
    """
    Checks if a user can view a project's settings based on their role
    and logs their access attempt.

    Args:
        uid: The user's unique identifier.
        role: The user's role (e.g., 'project_manager', 'team_member', 'viewer').
        proj: The project identifier.

    Raises:
        KeyError: If the project identifier is not found in PROJECTS_DB.
        PermissionError: If the user's role does not grant access to project settings.
    """
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "user_id": uid,
        "project_id": proj,
        "role": role,
        "action": "view_project_settings",
        "status": "" # To be filled based on outcome
    }

    if proj not in PROJECTS_DB:
        log_entry["status"] = "failure_project_not_found"
        ACCESS_LOG.append(log_entry)
        raise KeyError(f"Project {proj} not found.")

    allowed_roles_for_viewing = ['project_manager', 'team_member']
    can_view_settings = role in allowed_roles_for_viewing

    if can_view_settings:
        log_entry["status"] = "success_granted"
        ACCESS_LOG.append(log_entry)
        # Access granted, proceed (e.g., return settings or True, though prompt implies just logging and erroring)
        # For this implementation, successful access means the function completes without error.
        print(f"User {uid} granted access to settings for project {proj}.")
    else:
        log_entry["status"] = "failure_permission_denied"
        ACCESS_LOG.append(log_entry)
        raise PermissionError(
            f"User {uid} with role '{role}' is not authorized to view settings for project {proj}."
        )

if __name__ == '__main__':
    # Example Usage:
    print("Attempting valid access (project_manager):")
    try:
        log_project_access("USER1", "project_manager", "PROJ001")
    except (PermissionError, KeyError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting valid access (team_member):")
    try:
        log_project_access("USER2", "team_member", "PROJ001")
    except (PermissionError, KeyError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting unauthorized access (viewer):")
    try:
        log_project_access("USER3", "viewer", "PROJ001")
    except (PermissionError, KeyError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("Attempting access to non-existent project:")
    try:
        log_project_access("USER1", "project_manager", "PROJ003")
    except (PermissionError, KeyError) as e:
        print(f"Error: {e}")
    print("-" * 20)

    print("\nAccess Log:")
    for entry in ACCESS_LOG:
        print(entry)
