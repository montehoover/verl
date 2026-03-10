# A simple in-memory store for users and their roles
# In a real application, this would be a database or an external auth system.
_users_roles = {
    "USER1": "admin",    # Admin for PROJ001
    "USER2": "editor",   # Editor for PROJ001
    "USER3": "viewer",   # Viewer for PROJ001
    "USER4": "editor",   # Editor for PROJ002
    "USER5": "viewer",   # Viewer for PROJ002
    "user123": "admin",  # Existing generic admin
    "user456": "editor", # Existing generic editor
    "user789": "viewer", # Existing generic viewer
}

def authenticate_user(user_id: str, role: str) -> bool:
    """
    Authenticates a user based on their user_id and role.

    Args:
        user_id: The ID of the user.
        role: The role to check for the user.

    Returns:
        True if the user has the specified role, False otherwise.
    """
    if user_id in _users_roles and _users_roles[user_id] == role:
        return True
    return False

# Predefined rules for project settings
# For simplicity, let's define some allowed keys and their types/constraints
ALLOWED_SETTINGS = {
    "project_name": str,
    "status": str,  # Allowed values: "active", "inactive", "archived"
    "max_users": int, # Cannot be decreased
    "description": str,
}

VALID_STATUSES = ["active", "inactive", "archived"]

def validate_project_settings(current_settings: dict, settings_update: dict) -> bool:
    """
    Validates proposed project settings updates against predefined rules.

    Args:
        current_settings: The current settings of the project.
        settings_update: The proposed updates to the settings.

    Returns:
        True if the settings_update is valid, False otherwise.
    """
    if not settings_update:
        print("Validation Error: Settings update cannot be empty.")
        return False

    for key, value in settings_update.items():
        if key not in ALLOWED_SETTINGS:
            print(f"Validation Error: Setting '{key}' is not a recognized project setting.")
            return False

        expected_type = ALLOWED_SETTINGS[key]
        if not isinstance(value, expected_type):
            print(f"Validation Error: Setting '{key}' must be of type {expected_type.__name__}, got {type(value).__name__}.")
            return False

        if key == "status":
            if value not in VALID_STATUSES:
                print(f"Validation Error: Status '{value}' is invalid. Must be one of {VALID_STATUSES}.")
                return False
        
        if key == "max_users":
            if key in current_settings and value < current_settings[key]:
                print(f"Validation Error: 'max_users' cannot be decreased from {current_settings[key]} to {value}.")
                return False
            if value <= 0:
                print(f"Validation Error: 'max_users' must be a positive integer.")
                return False

    return True

PROJECTS_DB = {
    "PROJ001": {
        "project_id": "PROJ001",
        "creator_id": "USER1",
        "members": ["USER1", "USER2", "USER3"],
        "settings": {"visibility": "private", "deadline": "2023-12-31", "project_name": "Alpha Project", "status": "active", "max_users": 5}
    },
    "PROJ002": {
        "project_id": "PROJ002",
        "creator_id": "USER4",
        "members": ["USER4", "USER5"],
        "settings": {"visibility": "public", "deadline": "2023-10-15", "project_name": "Beta Project", "status": "inactive", "max_users": 10}
    }
}

def change_project_config(uid: str, role: str, proj_id: str, settings_update: dict) -> str:
    """
    Changes project settings if the user is authorized and settings are valid.

    Args:
        uid: The user ID attempting the change.
        role: The role of the user.
        proj_id: The ID of the project to update.
        settings_update: A dictionary of settings to update.

    Returns:
        A success message string if the update is successful.

    Raises:
        PermissionError: If authentication fails or the user is not authorized.
        ValueError: If the project ID is not found or settings_update is invalid.
    """
    if not authenticate_user(uid, role):
        raise PermissionError(f"User '{uid}' authentication failed or does not have role '{role}'.")

    project = PROJECTS_DB.get(proj_id)
    if not project:
        raise ValueError(f"Project '{proj_id}' not found.")

    # Authorization check
    if role == 'admin':
        # Admin can change any project
        pass
    elif role == 'editor':
        if uid not in project["members"]:
            raise PermissionError(f"User '{uid}' (editor) is not a member of project '{proj_id}' and cannot change its settings.")
    else: # Viewers or other roles
        raise PermissionError(f"User '{uid}' with role '{role}' is not permitted to change project settings.")

    current_settings = project["settings"]
    if not validate_project_settings(current_settings, settings_update):
        # validate_project_settings prints specific errors
        raise ValueError(f"Invalid settings update for project '{proj_id}'. Check validation messages above.")

    # Apply the update
    project["settings"].update(settings_update)
    return f"Project '{proj_id}' settings updated successfully by user '{uid}' ({role})."

if __name__ == '__main__':
    # Example Usage for authenticate_user
    print("--- User Authentication ---")
    print(f"Authenticating USER1 as admin: {authenticate_user('USER1', 'admin')}")
    print(f"Authenticating USER2 as editor: {authenticate_user('USER2', 'editor')}")
    print(f"Authenticating USER3 as viewer: {authenticate_user('USER3', 'viewer')}")
    print(f"Authenticating user123 as editor: {authenticate_user('user123', 'editor')}") # Test existing user
    print(f"Authenticating unknown_user as admin: {authenticate_user('unknown_user', 'admin')}")
    print(f"Authenticating USER1 as unknown_role: {authenticate_user('USER1', 'unknown_role')}")

    # Example Usage for validate_project_settings
    print("\n--- Project Settings Validation (examples from before) ---")
    current_project_settings = {
        "project_name": "Omega Project",
        "status": "active",
        "max_users": 10,
        "description": "A very important project."
    }

    valid_update_1 = {"status": "inactive"}
    print(f"\nValidating update 1: {valid_update_1}")
    print(f"Result: {validate_project_settings(current_project_settings, valid_update_1)}")

    valid_update_2 = {"max_users": 15, "description": "Updated description."}
    print(f"\nValidating update 2: {valid_update_2}")
    print(f"Result: {validate_project_settings(current_project_settings, valid_update_2)}")

    invalid_update_1 = {"status": "on_hold"} # Invalid status
    print(f"\nValidating invalid update 1: {invalid_update_1}")
    print(f"Result: {validate_project_settings(current_project_settings, invalid_update_1)}")

    invalid_update_2 = {"unknown_setting": "some_value"} # Unknown setting key
    print(f"\nValidating invalid update 2: {invalid_update_2}")
    print(f"Result: {validate_project_settings(current_project_settings, invalid_update_2)}")

    invalid_update_3 = {"max_users": 5} # Decreasing max_users
    print(f"\nValidating invalid update 3: {invalid_update_3}")
    print(f"Result: {validate_project_settings(current_project_settings, invalid_update_3)}")
    
    invalid_update_4 = {"max_users": "ten"} # Wrong type for max_users
    print(f"\nValidating invalid update 4: {invalid_update_4}")
    print(f"Result: {validate_project_settings(current_project_settings, invalid_update_4)}")

    invalid_update_5 = {} # Empty update
    print(f"\nValidating invalid update 5 (empty): {invalid_update_5}")
    print(f"Result: {validate_project_settings(current_project_settings, invalid_update_5)}")
    
    invalid_update_6 = {"max_users": 0} # max_users not positive
    print(f"\nValidating invalid update 6: {invalid_update_6}")
    print(f"Result: {validate_project_settings(current_project_settings, invalid_update_6)}")

    # Example Usage for change_project_config
    print("\n--- Change Project Configuration ---")

    # Scenario 1: Admin (USER1) updates PROJ001 - success
    try:
        print("\nScenario 1: Admin (USER1) updates PROJ001 (description)")
        update1 = {"description": "Alpha project new description by admin"}
        result = change_project_config("USER1", "admin", "PROJ001", update1)
        print(f"Success: {result}")
        print(f"PROJ001 new settings: {PROJECTS_DB['PROJ001']['settings']}")
    except (PermissionError, ValueError) as e:
        print(f"Error: {e}")

    # Scenario 2: Editor (USER2) updates PROJ001 (member of project) - success
    try:
        print("\nScenario 2: Editor (USER2) updates PROJ001 (max_users)")
        # Note: max_users in PROJ001 is 5. This update increases it.
        update2 = {"max_users": 7}
        result = change_project_config("USER2", "editor", "PROJ001", update2)
        print(f"Success: {result}")
        print(f"PROJ001 new settings: {PROJECTS_DB['PROJ001']['settings']}")
    except (PermissionError, ValueError) as e:
        print(f"Error: {e}")

    # Scenario 3: Editor (USER4) tries to update PROJ001 (not a member) - PermissionError
    try:
        print("\nScenario 3: Editor (USER4) tries to update PROJ001 (not a member)")
        update3 = {"description": "Attempt by non-member editor"}
        result = change_project_config("USER4", "editor", "PROJ001", update3)
        print(f"Success: {result}") # Should not reach here
    except (PermissionError, ValueError) as e:
        print(f"Error: {e}")

    # Scenario 4: Viewer (USER3) tries to update PROJ001 - PermissionError
    try:
        print("\nScenario 4: Viewer (USER3) tries to update PROJ001")
        update4 = {"description": "Attempt by viewer"}
        result = change_project_config("USER3", "viewer", "PROJ001", update4)
        print(f"Success: {result}") # Should not reach here
    except (PermissionError, ValueError) as e:
        print(f"Error: {e}")

    # Scenario 5: Admin (USER1) tries to update non-existent project PROJ003 - ValueError
    try:
        print("\nScenario 5: Admin (USER1) tries to update non-existent PROJ003")
        update5 = {"description": "For non-existent project"}
        result = change_project_config("USER1", "admin", "PROJ003", update5)
        print(f"Success: {result}") # Should not reach here
    except (PermissionError, ValueError) as e:
        print(f"Error: {e}")

    # Scenario 6: Admin (USER1) tries an invalid settings update (e.g., unknown setting key) - ValueError
    try:
        print("\nScenario 6: Admin (USER1) provides invalid setting key for PROJ002")
        update6 = {"new_feature_toggle": True} # This key is not in ALLOWED_SETTINGS
        result = change_project_config("USER1", "admin", "PROJ002", update6)
        print(f"Success: {result}") # Should not reach here
    except (PermissionError, ValueError) as e:
        print(f"Error: {e}")

    # Scenario 7: Admin (USER1) tries an invalid settings value (e.g., decreasing max_users) - ValueError
    try:
        print("\nScenario 7: Admin (USER1) tries to decrease max_users for PROJ002")
        # PROJ002 max_users is 10.
        update7 = {"max_users": 8} 
        result = change_project_config("USER1", "admin", "PROJ002", update7)
        print(f"Success: {result}") # Should not reach here
    except (PermissionError, ValueError) as e:
        print(f"Error: {e}")
        print(f"PROJ002 settings remain: {PROJECTS_DB['PROJ002']['settings']}")
        
    # Scenario 8: Unauthenticated user (bad role) - PermissionError
    try:
        print("\nScenario 8: USER1 tries with a role they don't have ('superadmin')")
        update8 = {"description": "Attempt with bad role"}
        result = change_project_config("USER1", "superadmin", "PROJ001", update8)
        print(f"Success: {result}") # Should not reach here
    except (PermissionError, ValueError) as e:
        print(f"Error: {e}")
