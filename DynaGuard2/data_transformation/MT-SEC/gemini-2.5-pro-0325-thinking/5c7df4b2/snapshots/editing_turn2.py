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

SUGGESTIONS_LOG = []

def adjust_project_params(user_token: str, role: str, project_ident: str, suggested_values: dict) -> dict:
    """
    Processes and logs user suggestions for project settings modifications based on role authorization.

    Args:
        user_token: Identifier for the user.
        role: User's role ('project_manager', 'team_member', 'viewer').
        project_ident: The project's ID.
        suggested_values: A dictionary with proposed changes, expected to be in the format:
                          {"settings": {"setting_to_change": "new_value", ...}}

    Returns:
        A dictionary representing the log entry for the suggestion.

    Raises:
        PermissionError: If access is denied due to project not found, insufficient
                         permissions for the role, or user not being a member where required.
        ValueError: If the role is invalid, suggested_values is empty or malformed,
                    or no specific settings changes are provided.
    """
    if project_ident not in PROJECTS_DB:
        raise PermissionError(f"Project '{project_ident}' not found.")

    if not suggested_values:
        raise ValueError("suggested_values cannot be empty when attempting to make suggestions.")

    project_data = PROJECTS_DB[project_ident]

    if "settings" not in suggested_values or not isinstance(suggested_values["settings"], dict):
        raise ValueError("Suggestions must be for project 'settings' and provided as a dictionary under the 'settings' key.")
    
    proposed_settings_changes = suggested_values["settings"]
    if not proposed_settings_changes: # e.g. suggested_values = {"settings": {}}
        raise ValueError("No specific settings changes provided in suggested_values.")

    if role == "project_manager":
        # Project managers can suggest any changes to settings.
        # No specific user_token check for PM role in this simplified model.
        pass # Allowed to proceed

    elif role == "team_member":
        if user_token not in project_data.get("members", []):
            raise PermissionError(f"User '{user_token}' is not a member of project '{project_ident}'.")
        
        # Team members can only suggest changes to 'deadline'.
        allowed_setting_keys_for_tm = {"deadline"}
        for key_to_change in proposed_settings_changes.keys():
            if key_to_change not in allowed_setting_keys_for_tm:
                raise PermissionError(
                    f"Team members can only suggest changes to the following settings: "
                    f"{', '.join(allowed_setting_keys_for_tm)}. Attempted to change '{key_to_change}'."
                )
        pass # Allowed to proceed

    elif role == "viewer":
        # Viewers cannot make any suggestions.
        # Since suggested_values and proposed_settings_changes are guaranteed to be non-empty here,
        # this path means a viewer is attempting to make a suggestion.
        raise PermissionError("Viewers are not authorized to make suggestions.")
        
    else:
        raise ValueError(f"Invalid role: {role}. Must be 'project_manager', 'team_member', or 'viewer'.")

    # If authorization passed, create and log the suggestion.
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_token": user_token,
        "role": role,
        "project_ident": project_ident,
        "suggested_changes": proposed_settings_changes 
    }
    SUGGESTIONS_LOG.append(log_entry)
    return log_entry
