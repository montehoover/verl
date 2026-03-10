# Global data stores (as per problem description)
USERS = {
    1: {"name": "Alice", "is_admin": True},
    2: {"name": "Bob", "is_admin": False},
    3: {"name": "Charlie", "is_admin": False},
}

PROJECTS = {
    101: {"name": "Project Alpha", "owner_id": 2},
    102: {"name": "Project Beta", "owner_id": 1},
    103: {"name": "Project Gamma", "owner_id": 3},
}

def log_project_membership_action(acting_user_id: int, prjt_id: int, member_action: str) -> bool:
    """
    Logs a project membership action ('add' or 'remove') if the acting user is the project owner.

    Args:
        acting_user_id: The ID of the user attempting the action.
        prjt_id: The ID of the project.
        member_action: The action to log ('add' or 'remove').

    Returns:
        True if the action was logged (user is owner), False otherwise.
    """
    user = USERS.get(acting_user_id)
    project = PROJECTS.get(prjt_id)

    if not user:
        # Optionally, log or raise an error if the user is not found
        # print(f"Error: User with ID {acting_user_id} not found.")
        return False

    if not project:
        # Optionally, log or raise an error if the project is not found
        # print(f"Error: Project with ID {prjt_id} not found.")
        return False

    # Check if the user is the owner of the project
    is_owner = (project.get("owner_id") == acting_user_id)

    if is_owner:
        if member_action not in ['add', 'remove']:
            # Optionally, log or raise an error for invalid action
            # print(f"Error: Invalid member_action '{member_action}'. Must be 'add' or 'remove'.")
            return False # Or handle as an error
        print(f"Action '{member_action}' for project {prjt_id} by owner {acting_user_id} logged successfully.")
        return True
    else:
        # User is not the owner, action not logged
        # print(f"User {acting_user_id} is not the owner of project {prjt_id}. Action '{member_action}' not logged.")
        return False

if __name__ == '__main__':
    # Example Usage:
    # Test case 1: Admin user (Alice, ID 1), not owner, trying to log action for Project Alpha (ID 101, owned by Bob)
    print(f"Test Case 1 (Admin, Not Owner): Alice (1) logging 'add' for Project Alpha (101)")
    action_logged = log_project_membership_action(acting_user_id=1, prjt_id=101, member_action='add')
    print(f"Action Logged: {action_logged}\n") # Expected: False

    # Test case 2: Owner user (Bob, ID 2) trying to log action for Project Alpha (ID 101, owned by Bob)
    print(f"Test Case 2 (Owner): Bob (2) logging 'add' for Project Alpha (101)")
    action_logged = log_project_membership_action(acting_user_id=2, prjt_id=101, member_action='add')
    print(f"Action Logged: {action_logged}\n") # Expected: True

    # Test case 3: Non-admin, non-owner user (Charlie, ID 3) trying to log action for Project Alpha (ID 101)
    print(f"Test Case 3 (Not Owner): Charlie (3) logging 'remove' for Project Alpha (101)")
    action_logged = log_project_membership_action(acting_user_id=3, prjt_id=101, member_action='remove')
    print(f"Action Logged: {action_logged}\n") # Expected: False

    # Test case 4: User (Bob, ID 2) trying to log action for a project they don't own (Project Beta, ID 102, owned by Alice)
    print(f"Test Case 4 (Not Owner): Bob (2) logging 'add' for Project Beta (102)")
    action_logged = log_project_membership_action(acting_user_id=2, prjt_id=102, member_action='add')
    print(f"Action Logged: {action_logged}\n") # Expected: False

    # Test case 5: Admin user (Alice, ID 1), who is also owner, logging action for Project Beta (ID 102)
    print(f"Test Case 5 (Admin and Owner): Alice (1) logging 'remove' for Project Beta (102)")
    action_logged = log_project_membership_action(acting_user_id=1, prjt_id=102, member_action='remove')
    print(f"Action Logged: {action_logged}\n") # Expected: True

    # Test case 6: Non-existent user
    print(f"Test Case 6 (Non-existent User): User 99 logging 'add' for Project Alpha (101)")
    action_logged = log_project_membership_action(acting_user_id=99, prjt_id=101, member_action='add')
    print(f"Action Logged: {action_logged}\n") # Expected: False

    # Test case 7: Non-existent project
    print(f"Test Case 7 (Non-existent Project): Alice (1) logging 'add' for Project 999")
    action_logged = log_project_membership_action(acting_user_id=1, prjt_id=999, member_action='add')
    print(f"Action Logged: {action_logged}\n") # Expected: False
    
    # Test case 8: Owner user (Bob, ID 2) with invalid member_action
    print(f"Test Case 8 (Owner, Invalid Action): Bob (2) logging 'modify' for Project Alpha (101)")
    action_logged = log_project_membership_action(acting_user_id=2, prjt_id=101, member_action='modify')
    print(f"Action Logged: {action_logged}\n") # Expected: False
