# Global data structure for user roles
# Example: {user_id: role}
USER_ROLES = {
    1: 'admin',
    2: 'user',
    3: 'admin',
    4: 'user',
}

def get_user_role(user_id: int) -> str:
    """
    Determines the role of a user based on their user_id.

    Args:
        user_id: The integer ID of the user.

    Returns:
        A string representing the user's role ('admin' or 'user').
        Defaults to 'user' if the user_id is not found or has no specific role.
    """
    role = USER_ROLES.get(user_id)
    if role == 'admin':
        return 'admin'
    # Default to 'user' if not found or not 'admin'
    return 'user'

if __name__ == '__main__':
    # Example usage:
    print(f"User 1 role: {get_user_role(1)}")
    print(f"User 2 role: {get_user_role(2)}")
    print(f"User 5 role: {get_user_role(5)}") # Test a user not in USER_ROLES
    
    # Example of a user that might exist but not be an admin
    USER_ROLES[6] = 'editor' 
    print(f"User 6 role: {get_user_role(6)}")
