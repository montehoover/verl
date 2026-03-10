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

def adjust_project_params(user_token: str, role: str, project_ident: str, filter_params: dict) -> dict:
    """
    Retrieves a project's details based on the user's role authorization.

    Args:
        user_token: Identifier for the user.
        role: User's role ('project_manager', 'team_member', 'viewer').
        project_ident: The project's ID.
        filter_params: A dictionary where keys are the names of project attributes to return.
                       Example: {"project_id": True, "settings": True}

    Returns:
        A dictionary containing the requested project details allowed for the role.

    Raises:
        PermissionError: If access is denied due to project not found, insufficient
                         permissions, or user not being a member where required.
        ValueError: If the role is invalid.
    """
    if project_ident not in PROJECTS_DB:
        raise PermissionError(f"Project '{project_ident}' not found.")

    project_data = PROJECTS_DB[project_ident]
    result = {}
    
    allowed_keys_for_role = set()

    if role == "project_manager":
        # Project managers are assumed to have full access if the project exists.
        # No specific check against user_token for manager role in this simplified model.
        allowed_keys_for_role = set(project_data.keys())
    
    elif role == "team_member":
        if user_token not in project_data.get("members", []):
            raise PermissionError(f"User '{user_token}' is not a member of project '{project_ident}'.")
        # Team members can access general project info and full settings.
        allowed_keys_for_role = {"project_id", "creator_id", "members", "settings"}

    elif role == "viewer":
        project_settings_data = project_data.get("settings", {})
        is_public = project_settings_data.get("visibility") == "public"
        is_member = user_token in project_data.get("members", [])
        
        if not is_public and not is_member:
            raise PermissionError(f"User '{user_token}' is not authorized to view private project '{project_ident}'.")
        # Viewers can access basic info and only 'visibility' from settings.
        allowed_keys_for_role = {"project_id", "creator_id", "settings"}
        
    else:
        raise ValueError(f"Invalid role: {role}. Must be 'project_manager', 'team_member', or 'viewer'.")

    # Construct the result based on filter_params and role permissions
    for requested_key in filter_params: # Iterate over keys requested by the user in filter_params
        if requested_key in project_data and requested_key in allowed_keys_for_role:
            if role == "viewer" and requested_key == "settings":
                # Viewer-specific handling for 'settings'
                current_project_settings = project_data.get("settings", {})
                result["settings"] = {} # Initialize with an empty dict for the "settings" key
                if "visibility" in current_project_settings:
                    result["settings"]["visibility"] = current_project_settings["visibility"]
            else:
                # For project_manager, team_member, or non-'settings' keys for viewer
                result[requested_key] = project_data[requested_key]
                
    return result
