def authenticate_user(user_id: str, role: str) -> bool:
    """
    Authenticate a user based on their user_id and role.
    
    Args:
        user_id: The unique identifier of the user
        role: The role of the user (e.g., 'admin', 'manager', 'member')
    
    Returns:
        bool: True if the user has access rights, False otherwise
    """
    # Define valid users and their roles
    # In a real application, this would query a database
    valid_users = {
        'user001': ['admin', 'manager', 'member'],
        'user002': ['manager', 'member'],
        'user003': ['member'],
        'user004': ['admin'],
        'user005': ['manager', 'member']
    }
    
    # Define valid roles in the system
    valid_roles = {'admin', 'manager', 'member'}
    
    # Check if the role is valid
    if role not in valid_roles:
        return False
    
    # Check if user exists and has the specified role
    if user_id in valid_users:
        return role in valid_users[user_id]
    
    return False


def validate_project_settings(current_settings: dict, settings_update: dict) -> bool:
    """
    Validate project settings update against predefined rules.
    
    Args:
        current_settings: The current project settings
        settings_update: The proposed settings update
    
    Returns:
        bool: True if the settings update is valid, False otherwise
    """
    # Define validation rules
    REQUIRED_FIELDS = {'project_name', 'project_id', 'status'}
    VALID_STATUSES = {'active', 'paused', 'completed', 'archived'}
    MAX_NAME_LENGTH = 100
    MAX_DESCRIPTION_LENGTH = 500
    MAX_TEAM_SIZE = 50
    VALID_PRIORITIES = {'low', 'medium', 'high', 'critical'}
    
    # Merge current settings with updates to get final state
    final_settings = current_settings.copy()
    final_settings.update(settings_update)
    
    # Check required fields are present
    for field in REQUIRED_FIELDS:
        if field not in final_settings or final_settings[field] is None:
            return False
    
    # Validate project_name
    if 'project_name' in settings_update:
        name = settings_update['project_name']
        if not isinstance(name, str) or len(name) == 0 or len(name) > MAX_NAME_LENGTH:
            return False
    
    # Validate project_id (should not be changed)
    if 'project_id' in settings_update:
        if current_settings.get('project_id') and settings_update['project_id'] != current_settings['project_id']:
            return False
    
    # Validate status
    if 'status' in settings_update:
        if settings_update['status'] not in VALID_STATUSES:
            return False
    
    # Validate description
    if 'description' in settings_update:
        desc = settings_update['description']
        if desc is not None and (not isinstance(desc, str) or len(desc) > MAX_DESCRIPTION_LENGTH):
            return False
    
    # Validate team_size
    if 'team_size' in settings_update:
        size = settings_update['team_size']
        if not isinstance(size, int) or size < 1 or size > MAX_TEAM_SIZE:
            return False
    
    # Validate priority
    if 'priority' in settings_update:
        if settings_update['priority'] not in VALID_PRIORITIES:
            return False
    
    # Validate deadline (should be a string in ISO format)
    if 'deadline' in settings_update:
        deadline = settings_update['deadline']
        if deadline is not None:
            if not isinstance(deadline, str):
                return False
            try:
                from datetime import datetime
                datetime.fromisoformat(deadline.replace('Z', '+00:00'))
            except:
                return False
    
    # Validate budget (should be positive)
    if 'budget' in settings_update:
        budget = settings_update['budget']
        if budget is not None:
            if not isinstance(budget, (int, float)) or budget < 0:
                return False
    
    return True


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


def change_project_config(uid: str, role: str, proj: str, settings_update: dict) -> str:
    """
    Change project configuration if the user is authorized.
    
    Args:
        uid: User ID
        role: User role
        proj: Project ID
        settings_update: Dictionary of settings to update
        
    Returns:
        str: Success message if update is successful
        
    Raises:
        PermissionError: If user is not authorized or update is invalid
    """
    # Check if project exists
    if proj not in PROJECTS_DB:
        raise PermissionError(f"Project {proj} does not exist")
    
    project = PROJECTS_DB[proj]
    
    # Check if user is a member of the project
    if uid not in project["members"]:
        raise PermissionError(f"User {uid} is not a member of project {proj}")
    
    # Check if user has valid authentication
    if not authenticate_user(uid, role):
        raise PermissionError(f"User {uid} does not have role {role}")
    
    # Check if role has permission to modify settings
    # Only admin and manager roles can modify project settings
    if role not in ['admin', 'manager']:
        raise PermissionError(f"Role {role} does not have permission to modify project settings")
    
    # For managers, additional check - they must be the creator or have admin role
    if role == 'manager' and uid != project["creator_id"]:
        raise PermissionError(f"Manager {uid} can only modify projects they created")
    
    # Validate the settings update
    current_settings = project["settings"].copy()
    # Add required fields for validation
    current_settings.update({
        "project_id": proj,
        "project_name": f"Project {proj}",
        "status": "active"
    })
    
    if not validate_project_settings(current_settings, settings_update):
        raise PermissionError("Invalid settings update")
    
    # Apply the settings update
    project["settings"].update(settings_update)
    
    return f"Successfully updated settings for project {proj}"


# Example usage
if __name__ == "__main__":
    # Test cases for authenticate_user
    test_cases = [
        ("user001", "admin"),      # True
        ("user001", "member"),     # True
        ("user002", "admin"),      # False
        ("user003", "member"),     # True
        ("user999", "member"),     # False
        ("user001", "invalid"),    # False
    ]
    
    for user_id, role in test_cases:
        result = authenticate_user(user_id, role)
        print(f"authenticate_user('{user_id}', '{role}') = {result}")
    
    print("\n" + "="*50 + "\n")
    
    # Test cases for validate_project_settings
    current = {
        "project_id": "proj_123",
        "project_name": "Website Redesign",
        "status": "active",
        "team_size": 5,
        "priority": "high"
    }
    
    test_updates = [
        # Valid updates
        {"status": "paused"},
        {"project_name": "New Website Design", "priority": "critical"},
        {"team_size": 10, "description": "Updated project scope"},
        {"deadline": "2024-12-31T23:59:59Z", "budget": 50000},
        
        # Invalid updates
        {"project_id": "proj_456"},  # Can't change ID
        {"status": "cancelled"},  # Invalid status
        {"team_size": 100},  # Too large
        {"priority": "urgent"},  # Invalid priority
        {"project_name": ""},  # Empty name
        {"budget": -1000},  # Negative budget
    ]
    
    for update in test_updates:
        result = validate_project_settings(current, update)
        print(f"validate_project_settings(current, {update}) = {result}")
    
    print("\n" + "="*50 + "\n")
    
    # Test cases for change_project_config
    # Update valid_users to match PROJECTS_DB users
    valid_users_backup = {
        'user001': ['admin', 'manager', 'member'],
        'user002': ['manager', 'member'],
        'user003': ['member'],
        'user004': ['admin'],
        'user005': ['manager', 'member']
    }
    
    # Temporarily update authenticate_user's valid_users for testing
    import sys
    module = sys.modules[__name__]
    
    # Define test valid users matching PROJECTS_DB
    test_valid_users = {
        'USER1': ['admin', 'manager'],
        'USER2': ['member'],
        'USER3': ['manager'],
        'USER4': ['admin'],
        'USER5': ['member']
    }
    
    # Override authenticate_user for testing
    def test_authenticate_user(user_id: str, role: str) -> bool:
        valid_roles = {'admin', 'manager', 'member'}
        if role not in valid_roles:
            return False
        if user_id in test_valid_users:
            return role in test_valid_users[user_id]
        return False
    
    # Temporarily replace authenticate_user
    original_authenticate = authenticate_user
    module.authenticate_user = test_authenticate_user
    
    config_test_cases = [
        # Successful cases
        ("USER1", "admin", "PROJ001", {"visibility": "public"}, True),
        ("USER1", "manager", "PROJ001", {"deadline": "2024-06-30"}, True),
        ("USER4", "admin", "PROJ002", {"visibility": "private"}, True),
        
        # Permission errors
        ("USER2", "member", "PROJ001", {"visibility": "public"}, False),  # member can't modify
        ("USER3", "manager", "PROJ001", {"deadline": "2024-01-01"}, False),  # manager not creator
        ("USER5", "member", "PROJ002", {"visibility": "private"}, False),  # member can't modify
        ("USER1", "admin", "PROJ003", {"visibility": "public"}, False),  # project doesn't exist
        ("USER6", "admin", "PROJ001", {"visibility": "public"}, False),  # user not member
    ]
    
    for uid, role, proj, update, should_succeed in config_test_cases:
        try:
            result = change_project_config(uid, role, proj, update)
            if should_succeed:
                print(f"✓ change_project_config('{uid}', '{role}', '{proj}', {update}) = {result}")
            else:
                print(f"✗ change_project_config('{uid}', '{role}', '{proj}', {update}) = Unexpected success")
        except PermissionError as e:
            if not should_succeed:
                print(f"✓ change_project_config('{uid}', '{role}', '{proj}', {update}) = PermissionError: {e}")
            else:
                print(f"✗ change_project_config('{uid}', '{role}', '{proj}', {update}) = Unexpected PermissionError: {e}")
    
    # Restore original authenticate_user
    module.authenticate_user = original_authenticate
