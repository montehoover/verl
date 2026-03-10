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

def adjust_project_params(user_token: str, role: str, project_ident: str, updated_values: dict) -> str:
    """
    Updates project settings based on user role and creator status.

    'project_managers' or the project's 'creator_id' can update settings.
    'team_members' and 'viewers' are restricted from making changes.

    Args:
        user_token: Identifier for the user.
        role: User's role ('project_manager', 'team_member', 'viewer').
        project_ident: The project's ID.
        updated_values: A dictionary with changes to be applied, expected to be in the format:
                          {"settings": {"setting_to_change": "new_value", ...}}

    Returns:
        A success message string if the update is allowed.

    Raises:
        PermissionError: If access is denied due to project not found, or insufficient
                         permissions for the role/user.
        ValueError: If the role is invalid, updated_values is empty or malformed,
                    or no specific settings changes are provided.
    """
    if project_ident not in PROJECTS_DB:
        raise PermissionError(f"Project '{project_ident}' not found.")

    if not updated_values:
        raise ValueError("updated_values cannot be empty when attempting to make updates.")

    # Ensure 'settings' key exists and its value is a dictionary in updated_values
    if "settings" not in updated_values or not isinstance(updated_values.get("settings"), dict):
        raise ValueError("Updates must be for project 'settings' and provided as a dictionary under the 'settings' key.")
    
    settings_to_update = updated_values["settings"]
    if not settings_to_update: # e.g., updated_values = {"settings": {}}
        raise ValueError("No specific settings changes provided in updated_values.")

    project_data = PROJECTS_DB[project_ident] # project_ident is confirmed to be in PROJECTS_DB
    
    valid_roles = {"project_manager", "team_member", "viewer"}
    if role not in valid_roles:
        # Sort for consistent error message
        raise ValueError(f"Invalid role: '{role}'. Must be one of {', '.join(sorted(list(valid_roles)))}.")

    is_creator = user_token == project_data.get("creator_id")
    is_project_manager = role == "project_manager"

    if is_project_manager or is_creator:
        # Authorized to update
        # Ensure 'settings' key exists in project_data, though it should from DB structure
        if "settings" not in project_data: 
            project_data["settings"] = {} # Defensive coding, though unlikely with current DB
        project_data["settings"].update(settings_to_update)
        return f"Project '{project_ident}' settings updated successfully."
    else:
        # If not a project_manager and not the creator,
        # and role is one of the valid roles ('team_member' or 'viewer' at this point),
        # then they are not authorized.
        raise PermissionError(f"User '{user_token}' with role '{role}' is not authorized to update project settings for '{project_ident}'.")
