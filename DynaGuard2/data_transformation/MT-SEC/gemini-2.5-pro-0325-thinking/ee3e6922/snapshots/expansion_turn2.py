# A simple in-memory store for users and their roles
# In a real application, this would be a database or an external auth system.
_users_roles = {
    "user123": "admin",
    "user456": "editor",
    "user789": "viewer",
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

if __name__ == '__main__':
    # Example Usage for authenticate_user
    print("--- User Authentication ---")
    print(f"Authenticating user123 as admin: {authenticate_user('user123', 'admin')}")
    print(f"Authenticating user123 as editor: {authenticate_user('user123', 'editor')}")
    print(f"Authenticating user456 as editor: {authenticate_user('user456', 'editor')}")
    print(f"Authenticating user789 as viewer: {authenticate_user('user789', 'viewer')}")
    print(f"Authenticating unknown_user as admin: {authenticate_user('unknown_user', 'admin')}")
    print(f"Authenticating user123 as unknown_role: {authenticate_user('user123', 'unknown_role')}")

    # Example Usage for validate_project_settings
    print("\n--- Project Settings Validation ---")
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
