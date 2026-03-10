# Global data structure for user roles
# Example: {user_id: role}
USER_ROLES = {
    1: 'admin',
    2: 'user',
    3: 'admin',
    4: 'user',
}

# Global data structure for project memberships
# Example: {proj_id: set_of_user_ids}
PROJECT_MEMBERSHIPS = {
    101: {1, 2},
    102: {2, 3},
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

def modify_project_membership(proj_id: int, operation: str, user_id: int) -> bool:
    """
    Modifies the membership of a user in a project.

    Args:
        proj_id: The integer ID of the project.
        operation: The operation to perform ('add' or 'remove').
        user_id: The integer ID of the user.

    Returns:
        True if the operation was successful, False otherwise.
    """
    if operation == 'add':
        if proj_id not in PROJECT_MEMBERSHIPS:
            PROJECT_MEMBERSHIPS[proj_id] = set()
        if user_id in PROJECT_MEMBERSHIPS[proj_id]: # Already a member
            return True # Or False if adding an existing member is a failure to change state
        PROJECT_MEMBERSHIPS[proj_id].add(user_id)
        return True
    elif operation == 'remove':
        if proj_id in PROJECT_MEMBERSHIPS and user_id in PROJECT_MEMBERSHIPS[proj_id]:
            PROJECT_MEMBERSHIPS[proj_id].remove(user_id)
            # If the project becomes empty, we could optionally remove it from PROJECT_MEMBERSHIPS
            # if not PROJECT_MEMBERSHIPS[proj_id]:
            #     del PROJECT_MEMBERSHIPS[proj_id]
            return True
        return False # User or project not found, or user not in project
    else:
        # Invalid operation
        return False

if __name__ == '__main__':
    # Example usage for get_user_role:
    print(f"User 1 role: {get_user_role(1)}")
    print(f"User 2 role: {get_user_role(2)}")
    print(f"User 5 role: {get_user_role(5)}") # Test a user not in USER_ROLES
    USER_ROLES[6] = 'editor'
    print(f"User 6 role: {get_user_role(6)}")

    print("\n--- Project Membership Modification ---")
    # Initial state
    print(f"Initial project memberships: {PROJECT_MEMBERSHIPS}")

    # Add user 4 to project 101
    print(f"Adding user 4 to project 101: {modify_project_membership(101, 'add', 4)}")
    print(f"Project 101 members: {PROJECT_MEMBERSHIPS.get(101)}")

    # Add user 1 to project 103 (new project)
    print(f"Adding user 1 to project 103: {modify_project_membership(103, 'add', 1)}")
    print(f"Project 103 members: {PROJECT_MEMBERSHIPS.get(103)}")
    print(f"All project memberships: {PROJECT_MEMBERSHIPS}")

    # Remove user 2 from project 101
    print(f"Removing user 2 from project 101: {modify_project_membership(101, 'remove', 2)}")
    print(f"Project 101 members: {PROJECT_MEMBERSHIPS.get(101)}")

    # Try to remove user 5 (non-member) from project 101
    print(f"Removing user 5 from project 101: {modify_project_membership(101, 'remove', 5)}")
    print(f"Project 101 members: {PROJECT_MEMBERSHIPS.get(101)}")

    # Try to remove user 1 from project 999 (non-existent project)
    print(f"Removing user 1 from project 999: {modify_project_membership(999, 'remove', 1)}")

    # Invalid operation
    print(f"Invalid operation 'delete' for user 1 in project 101: {modify_project_membership(101, 'delete', 1)}")
    print(f"All project memberships: {PROJECT_MEMBERSHIPS}")
