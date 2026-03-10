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

def check_user_permission(user_id: int, project_id: int) -> bool:
    """
    Checks if a user has permission to manage a project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.

    Returns:
        True if the user can manage the project, False otherwise.
    """
    user = USERS.get(user_id)
    project = PROJECTS.get(project_id)

    if not user:
        # User not found
        return False

    if not project:
        # Project not found
        return False

    # Admins can manage any project
    if user["role"] == "admin":
        return True

    # Project owners can manage their projects
    if project["owner_id"] == user_id:
        return True

    return False

def authorize_user_action(acting_user_id: int, project_identifier: int, modification: str) -> bool:
    """
    Authorizes a user to perform an action on a project.

    Args:
        acting_user_id: The ID of the user performing the action.
        project_identifier: The ID of the project.
        modification: A string describing the action (e.g., "add_member", "remove_member").
                      This is not strictly used in the current logic but included for future enhancements.

    Returns:
        True if the user is authorized, False otherwise.
    """
    # For now, any user who can manage the project can perform any action.
    # The 'modification' parameter can be used later for more granular control.
    return check_user_permission(acting_user_id, project_identifier)

def update_project_members(acting_user_id: int, project_identifier: int, modification: str, user_to_update_id: int) -> bool:
    """
    Updates project membership (adds or removes a user).

    Args:
        acting_user_id: The ID of the user performing the action.
        project_identifier: The ID of the project.
        modification: The action to perform ("add_member" or "remove_member").
        user_to_update_id: The ID of the user whose membership is to be updated.

    Returns:
        True if the update was successful, False otherwise.
    """
    if not authorize_user_action(acting_user_id, project_identifier, modification):
        print(f"Authorization failed for user {acting_user_id} on project {project_identifier} for action {modification}.")
        return False

    project = PROJECTS.get(project_identifier)
    if not project:
        print(f"Project {project_identifier} not found.")
        return False

    user_to_update = USERS.get(user_to_update_id)
    if not user_to_update:
        print(f"User to update {user_to_update_id} not found.")
        return False

    if modification == "add_member":
        if user_to_update_id not in project["members"]:
            project["members"].append(user_to_update_id)
            print(f"User {user_to_update_id} added to project {project_identifier}.")
            return True
        else:
            print(f"User {user_to_update_id} is already a member of project {project_identifier}.")
            return False # Or True, depending on desired idempotency behavior
    elif modification == "remove_member":
        if user_to_update_id == project["owner_id"]:
            print(f"Cannot remove project owner (user {user_to_update_id}) from project {project_identifier}.")
            return False
        if user_to_update_id in project["members"]:
            project["members"].remove(user_to_update_id)
            print(f"User {user_to_update_id} removed from project {project_identifier}.")
            return True
        else:
            print(f"User {user_to_update_id} is not a member of project {project_identifier}.")
            return False
    else:
        print(f"Invalid modification action: {modification}.")
        return False

if __name__ == '__main__':
    # Test cases for check_user_permission
    print(f"Admin (1) managing project 101: {check_user_permission(1, 101)}")  # Expected: True
    print(f"Owner (2) managing project 101: {check_user_permission(2, 101)}")  # Expected: True
    print(f"Non-owner (3) managing project 101: {check_user_permission(3, 101)}") # Expected: False
    print(f"Owner (3) managing project 102: {check_user_permission(3, 102)}")  # Expected: True
    print(f"User (2) managing non-existent project 999: {check_user_permission(2, 999)}") # Expected: False
    print(f"Non-existent user 99 managing project 101: {check_user_permission(99, 101)}") # Expected: False
    print(f"Admin (1) managing non-existent project 999: {check_user_permission(1, 999)}") # Expected: False

    print("\n# Test cases for authorize_user_action")
    # Admin (1) tries to add a member to project 101
    print(f"Admin (1) action 'add_member' on project 101: {authorize_user_action(1, 101, 'add_member')}")  # Expected: True
    # Owner (2) tries to remove a member from project 101
    print(f"Owner (2) action 'remove_member' on project 101: {authorize_user_action(2, 101, 'remove_member')}")  # Expected: True
    # Non-owner (3) tries to add a member to project 101
    print(f"Non-owner (3) action 'add_member' on project 101: {authorize_user_action(3, 101, 'add_member')}") # Expected: False
    # Owner (3) tries to modify project 102
    print(f"Owner (3) action 'modify_settings' on project 102: {authorize_user_action(3, 102, 'modify_settings')}")  # Expected: True
    # User (2) tries an action on a non-existent project 999
    print(f"User (2) action 'add_member' on non-existent project 999: {authorize_user_action(2, 999, 'add_member')}") # Expected: False
    # Non-existent user 99 tries an action on project 101
    print(f"Non-existent user (99) action 'add_member' on project 101: {authorize_user_action(99, 101, 'add_member')}") # Expected: False

    print("\n# Test cases for update_project_members")
    # Admin (1) adds user 4 to project 101
    print(f"Admin (1) adds user 4 to project 101: {update_project_members(1, 101, 'add_member', 4)}")  # Expected: True
    print(f"Project 101 members after admin add: {PROJECTS[101]['members']}") # Expected: [2, 3, 4]

    # Owner (2) removes user 3 from project 101
    print(f"Owner (2) removes user 3 from project 101: {update_project_members(2, 101, 'remove_member', 3)}")  # Expected: True
    print(f"Project 101 members after owner remove: {PROJECTS[101]['members']}") # Expected: [2, 4]

    # Non-owner (3) tries to add user 2 to project 102 (should fail auth)
    print(f"Non-owner (3) tries to add user 2 to project 102: {update_project_members(3, 102, 'add_member', 2)}") # Expected: False (user 3 is owner, but let's test with a non-owner trying to act on a project they don't own)
    # Correction: User 3 IS the owner of project 102. Let's use user 2 trying to modify project 102.
    print(f"Non-owner (2) tries to add user 1 to project 102: {update_project_members(2, 102, 'add_member', 1)}") # Expected: False
    print(f"Project 102 members (should be unchanged): {PROJECTS[102]['members']}") # Expected: [3, 4]

    # Owner (3) tries to remove self (owner) from project 102 (should fail)
    print(f"Owner (3) tries to remove self (3) from project 102: {update_project_members(3, 102, 'remove_member', 3)}")  # Expected: False
    print(f"Project 102 members (should be unchanged): {PROJECTS[102]['members']}") # Expected: [3, 4]

    # Admin (1) removes owner (4) from project 103 (should fail, as per current logic)
    # Note: The current logic prevents removing the owner. If admins should bypass this, the rule needs adjustment.
    print(f"Admin (1) tries to remove owner (4) from project 103: {update_project_members(1, 103, 'remove_member', 4)}")  # Expected: False
    print(f"Project 103 members (should be unchanged): {PROJECTS[103]['members']}") # Expected: [4]

    # Owner (2) tries to add non-existent user 99 to project 101 (should fail)
    print(f"Owner (2) tries to add non-existent user 99 to project 101: {update_project_members(2, 101, 'add_member', 99)}")  # Expected: False
    print(f"Project 101 members (should be unchanged): {PROJECTS[101]['members']}") # Expected: [2, 4]

    # Admin (1) tries to add user 2 to non-existent project 999 (should fail)
    print(f"Admin (1) tries to add user 2 to non-existent project 999: {update_project_members(1, 999, 'add_member', 2)}")  # Expected: False

    # Owner (2) tries to remove user 3 (already removed) from project 101 (should fail)
    print(f"Owner (2) tries to remove user 3 (already removed) from project 101: {update_project_members(2, 101, 'remove_member', 3)}")  # Expected: False
    print(f"Project 101 members (should be unchanged): {PROJECTS[101]['members']}") # Expected: [2, 4]

    # Owner (2) tries to add user 4 (already member) to project 101 (should fail)
    print(f"Owner (2) tries to add user 4 (already member) to project 101: {update_project_members(2, 101, 'add_member', 4)}")  # Expected: False
    print(f"Project 101 members (should be unchanged): {PROJECTS[101]['members']}") # Expected: [2, 4]
