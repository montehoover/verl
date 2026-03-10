USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

def has_permission(user_id: int, proj_id: int) -> bool:
    """
    Determines if a user has the right to manage a project.
    Admins can manage any project.
    Users can manage projects they own.
    """
    user = USERS.get(user_id)
    project = PROJECTS.get(proj_id)

    if not user:
        # User not found
        return False
    
    if not project:
        # Project not found
        return False

    if user["role"] == "admin":
        return True
    
    if project["owner_id"] == user_id:
        return True
        
    return False

def can_perform_action(acting_user_id: int, prj_id: int, action_type: str) -> bool:
    """
    Determines if a user can perform a specific action on a project.
    For now, any action is allowed if the user has management permission.
    """
    # Future: action_type could be 'add_member', 'remove_member', 'delete_project', etc.
    # and specific logic could be added here based on the action_type.
    return has_permission(acting_user_id, prj_id)

if __name__ == '__main__':
    # Test cases
    print(f"Admin (1) managing project 101: {has_permission(1, 101)}")  # Expected: True
    print(f"User 2 (owner) managing project 101: {has_permission(2, 101)}")  # Expected: True
    print(f"User 3 (not owner) managing project 101: {has_permission(3, 101)}")  # Expected: False
    print(f"User 2 (owner) managing project 102: {has_permission(2, 102)}")  # Expected: False
    print(f"User 5 (non-existent) managing project 101: {has_permission(5, 101)}") # Expected: False
    print(f"User 2 managing project 999 (non-existent): {has_permission(2, 999)}") # Expected: False
    print(f"Admin (1) managing project 999 (non-existent): {has_permission(1, 999)}") # Expected: False

    print("\n-- Action Permission Tests --")
    # Test cases for can_perform_action
    print(f"Admin (1) adding member to project 101: {can_perform_action(1, 101, 'add_member')}")  # Expected: True
    print(f"User 2 (owner) removing member from project 101: {can_perform_action(2, 101, 'remove_member')}")  # Expected: True
    print(f"User 3 (not owner) adding member to project 101: {can_perform_action(3, 101, 'add_member')}")  # Expected: False
    print(f"User 2 (owner) deleting project 102: {can_perform_action(2, 102, 'delete_project')}")  # Expected: False (not owner of 102)
    print(f"User 4 (owner) adding member to project 103: {can_perform_action(4, 103, 'add_member')}") # Expected: True
    print(f"User 3 (member, not owner) adding member to project 102: {can_perform_action(3, 102, 'add_member')}") # Expected: True (owner of 102)
    print(f"User 3 (owner of 102) adding member to project 101: {can_perform_action(3, 101, 'add_member')}") # Expected: False (not owner of 101)
    print(f"User 5 (non-existent) adding member to project 101: {can_perform_action(5, 101, 'add_member')}") # Expected: False
    print(f"User 2 adding member to project 999 (non-existent): {can_perform_action(2, 999, 'add_member')}") # Expected: False
