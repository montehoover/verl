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


# Example usage
if __name__ == "__main__":
    # Test cases
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
