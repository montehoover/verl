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

def get_user_role_and_project_owner(user_id: int, proj_id: int):
    """
    Determines a user's role and a project's owner.

    Args:
        user_id: The ID of the user.
        proj_id: The ID of the project.

    Returns:
        A tuple containing the user's role and the project's owner ID.
        Returns (None, None) if user or project information is not found.
    """
    user_info = USERS.get(user_id)
    project_info = PROJECTS.get(proj_id)

    user_role = user_info.get("role") if user_info else None
    project_owner_id = project_info.get("owner_id") if project_info else None

    return user_role, project_owner_id

def can_user_perform_action(active_user_id: int, proj_id: int, operation: str) -> bool:
    """
    Checks if a user can perform a specific action on a project.

    Args:
        active_user_id: The ID of the user performing the action.
        proj_id: The ID of the project.
        operation: The action to perform (e.g., "view", "edit", "delete").

    Returns:
        True if the user can perform the action, False otherwise.
    """
    user_role, project_owner_id = get_user_role_and_project_owner(active_user_id, proj_id)

    if not user_role or not project_owner_id: # User or project not found
        return False

    if user_role == "admin":
        return True  # Admins can do anything

    project_info = PROJECTS.get(proj_id)
    if not project_info: # Should not happen if project_owner_id is valid, but good practice
        return False

    is_owner = (active_user_id == project_owner_id)
    is_member = active_user_id in project_info.get("members", [])

    if operation == "view":
        return is_member # Members (including owner) can view
    elif operation == "edit":
        return is_owner # Only owners can edit
    elif operation == "delete":
        return is_owner # Only owners can delete
    
    return False # Unknown operation or insufficient permissions

def handle_project_membership(active_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Manages project memberships (add/remove members).
    Admins can manage memberships for any project.
    Project owners can manage memberships for their own projects.

    Args:
        active_user_id: The ID of the user performing the action.
        proj_id: The ID of the project.
        operation: "add" or "remove".
        target_member_id: The ID of the user to be added or removed.

    Returns:
        True if the action was successful, False otherwise.
    """
    # Validate target_member_id exists in USERS
    if target_member_id not in USERS:
        return False

    # Validate proj_id exists in PROJECTS
    project_info = PROJECTS.get(proj_id)
    if not project_info:
        return False

    # Validate active_user_id exists and get their role
    active_user_details = USERS.get(active_user_id)
    if not active_user_details:
        return False
    active_user_role = active_user_details.get("role")

    # Determine the actual owner of the project
    actual_project_owner_id = project_info.get("owner_id")

    # Check permissions: Admin or Owner of this specific project
    can_manage_members = False
    if active_user_role == "admin":
        can_manage_members = True
    elif active_user_id == actual_project_owner_id:
        can_manage_members = True
    
    if not can_manage_members:
        return False

    # Perform the operation
    members_list = project_info["members"]

    if operation == "add":
        if target_member_id not in members_list:
            members_list.append(target_member_id)
            return True
        else: # Member already exists or target user does not exist
            return False
    elif operation == "remove":
        if target_member_id in members_list:
            members_list.remove(target_member_id)
            return True
        else: # Member not found in project
            return False
    else: # Invalid operation string
        return False

if __name__ == '__main__':
    # Example usage for get_user_role_and_project_owner:
    role, owner = get_user_role_and_project_owner(1, 101)
    print(f"User 1, Project 101: Role = {role}, Owner ID = {owner}")

    role, owner = get_user_role_and_project_owner(3, 102)
    print(f"User 3, Project 102: Role = {role}, Owner ID = {owner}")

    role, owner = get_user_role_and_project_owner(5, 101) # Non-existent user
    print(f"User 5, Project 101: Role = {role}, Owner ID = {owner}")

    role, owner = get_user_role_and_project_owner(2, 200) # Non-existent project
    print(f"User 2, Project 200: Role = {role}, Owner ID = {owner}")

    print("\n# Example usage for can_user_perform_action:")
    # Admin user (user 1)
    print(f"Admin (1) can view project 101: {can_user_perform_action(1, 101, 'view')}")
    print(f"Admin (1) can edit project 101: {can_user_perform_action(1, 101, 'edit')}")
    print(f"Admin (1) can delete project 101: {can_user_perform_action(1, 101, 'delete')}")

    # Project owner (user 2 owns project 101)
    print(f"Owner (2) can view project 101: {can_user_perform_action(2, 101, 'view')}")
    print(f"Owner (2) can edit project 101: {can_user_perform_action(2, 101, 'edit')}")
    print(f"Owner (2) can delete project 101: {can_user_perform_action(2, 101, 'delete')}")

    # Project member (user 3 is a member of project 101, but not owner)
    print(f"Member (3) can view project 101: {can_user_perform_action(3, 101, 'view')}")
    print(f"Member (3) can edit project 101: {can_user_perform_action(3, 101, 'edit')}") # Should be False
    print(f"Member (3) can delete project 101: {can_user_perform_action(3, 101, 'delete')}") # Should be False

    # Non-member (user 4 is not a member of project 101)
    print(f"Non-member (4) can view project 101: {can_user_perform_action(4, 101, 'view')}") # Should be False
    print(f"Non-member (4) can edit project 101: {can_user_perform_action(4, 101, 'edit')}") # Should be False

    # Non-existent project
    print(f"User (2) can view non-existent project 200: {can_user_perform_action(2, 200, 'view')}") # Should be False

    # Non-existent user
    print(f"Non-existent user (5) can view project 101: {can_user_perform_action(5, 101, 'view')}") # Should be False

    print("\n# Example usage for handle_project_membership:")
    # Make a copy of members list for project 101 to reset between tests if needed
    original_p101_members = list(PROJECTS[101]['members'])
    original_p102_members = list(PROJECTS[102]['members'])

    print(f"Initial P101 members: {PROJECTS[101]['members']}")

    # Admin (user 1) adds user 4 to project 101
    print(f"Admin (1) adds user 4 to P101: {handle_project_membership(1, 101, 'add', 4)}") # True
    print(f"P101 members after admin add 4: {PROJECTS[101]['members']}") # Expected: [2, 3, 4]

    # Admin (user 1) removes user 3 from project 101
    print(f"Admin (1) removes user 3 from P101: {handle_project_membership(1, 101, 'remove', 3)}") # True
    print(f"P101 members after admin remove 3: {PROJECTS[101]['members']}") # Expected: [2, 4]

    # Reset P101 members for next set of tests
    PROJECTS[101]['members'] = list(original_p101_members)
    print(f"Reset P101 members: {PROJECTS[101]['members']}") # Expected: [2, 3]

    # Owner (user 2) of project 101 adds user 4 to project 101
    print(f"Owner (2) adds user 4 to P101: {handle_project_membership(2, 101, 'add', 4)}") # True
    print(f"P101 members after owner 2 adds 4: {PROJECTS[101]['members']}") # Expected: [2, 3, 4]

    # Owner (user 2) of project 101 removes user 3 from project 101
    print(f"Owner (2) removes user 3 from P101: {handle_project_membership(2, 101, 'remove', 3)}") # True
    print(f"P101 members after owner 2 removes 3: {PROJECTS[101]['members']}") # Expected: [2, 4]

    # Owner (user 2) of project 101 removes self (user 2) from project 101 members
    print(f"Owner (2) removes self (2) from P101 members: {handle_project_membership(2, 101, 'remove', 2)}") # True
    print(f"P101 members after owner 2 removes self: {PROJECTS[101]['members']}") # Expected: [4]
    
    # Reset P101 members
    PROJECTS[101]['members'] = list(original_p101_members)
    print(f"Reset P101 members: {PROJECTS[101]['members']}") # Expected: [2, 3]

    # Non-owner member (user 3) tries to add user 4 to project 101 (user 3 is member, not owner of P101)
    print(f"Non-owner member (3) tries to add user 4 to P101: {handle_project_membership(3, 101, 'add', 4)}") # False
    print(f"P101 members (should be unchanged): {PROJECTS[101]['members']}") # Expected: [2, 3]

    # Non-owner member (user 3) tries to remove user 2 from project 101
    print(f"Non-owner member (3) tries to remove user 2 from P101: {handle_project_membership(3, 101, 'remove', 2)}") # False
    print(f"P101 members (should be unchanged): {PROJECTS[101]['members']}") # Expected: [2, 3]

    # Attempt to add non-existent user (user 99) by admin (user 1)
    print(f"Admin (1) tries to add non-existent user (99) to P101: {handle_project_membership(1, 101, 'add', 99)}") # False
    print(f"P101 members (should be unchanged): {PROJECTS[101]['members']}") # Expected: [2, 3]

    # Attempt to add user 4 to non-existent project (project 999) by admin (user 1)
    print(f"Admin (1) tries to add user 4 to non-existent P999: {handle_project_membership(1, 999, 'add', 4)}") # False

    # Attempt by non-existent active user (user 98) to add user 4 to P101
    print(f"Non-existent active user (98) tries to add user 4 to P101: {handle_project_membership(98, 101, 'add', 4)}") # False
    
    # Attempt to add existing member (user 3) to P101 by owner (user 2)
    print(f"Owner (2) tries to add existing member (3) to P101: {handle_project_membership(2, 101, 'add', 3)}") # False
    print(f"P101 members (should be unchanged): {PROJECTS[101]['members']}") # Expected: [2, 3]

    # Attempt to remove non-member (user 4, after reset) from P101 by owner (user 2)
    print(f"Owner (2) tries to remove non-member (4) from P101: {handle_project_membership(2, 101, 'remove', 4)}") # False
    print(f"P101 members (should be unchanged): {PROJECTS[101]['members']}") # Expected: [2, 3]

    # Test with project 102 (owner is 3)
    PROJECTS[102]['members'] = list(original_p102_members) # Ensure P102 is reset
    print(f"Initial P102 members: {PROJECTS[102]['members']}") # Expected: [3, 4]
    # Owner of P102 (user 3) removes user 4
    print(f"Owner (3) removes user 4 from P102: {handle_project_membership(3, 102, 'remove', 4)}") # True
    print(f"P102 members: {PROJECTS[102]['members']}") # Expected: [3]
    # Owner of P102 (user 3) adds user 2
    print(f"Owner (3) adds user 2 to P102: {handle_project_membership(3, 102, 'add', 2)}") # True
    print(f"P102 members: {PROJECTS[102]['members']}") # Expected: [3, 2]
