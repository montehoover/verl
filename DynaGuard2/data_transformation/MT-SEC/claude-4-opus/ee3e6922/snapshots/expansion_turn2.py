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
