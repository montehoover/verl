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

def handle_project_membership(active_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Allows project owners to add or remove team members from their own projects,
    while admins can manage any project.

    Args:
        active_user_id: The ID of the user performing the task.
        proj_id: The ID of the project.
        operation: The operation to perform ('add' or 'remove').
        target_member_id: The ID of the user to be added or removed.

    Returns:
        True if the modification is successful, False otherwise.
    """
    active_user = USERS.get(active_user_id)
    project = PROJECTS.get(proj_id)
    target_user = USERS.get(target_member_id)

    if not active_user or not project or not target_user:
        # Invalid user, project, or target user ID
        return False

    is_admin = active_user["role"] == "admin"
    is_owner = project["owner_id"] == active_user_id

    if not (is_admin or is_owner):
        # User is not authorized to modify this project
        return False

    if operation == 'add':
        if target_member_id not in project["members"]:
            project["members"].append(target_member_id)
            return True
        return True # Already a member, consider it a success or no-op
    elif operation == 'remove':
        # Prevent owner from removing themselves if they are the last member and not an admin
        # Or, more simply, an owner cannot remove themselves.
        # This logic can be adjusted based on specific business rules.
        # For now, let's assume an owner cannot remove themselves directly via this function.
        # An admin could, or ownership would need to be transferred first.
        if target_member_id == project["owner_id"] and target_member_id == active_user_id and not is_admin:
             # Owner trying to remove themselves, not allowed unless admin is doing it
             # Or if we want to allow self-removal, this check changes.
             # For now, let's say owner cannot remove self.
            return False


        if target_member_id in project["members"]:
            project["members"].remove(target_member_id)
            return True
        return False # Target member not in project
    else:
        # Invalid operation
        return False

if __name__ == '__main__':
    print("Initial state:")
    print("USERS:", USERS)
    print("PROJECTS:", PROJECTS)
    print("-" * 30)

    # Test cases
    # Admin (user 1) adds user 4 to project 101 (owned by user 2)
    print("Admin (1) adds user 4 to project 101:")
    result = handle_project_membership(active_user_id=1, proj_id=101, operation='add', target_member_id=4)
    print(f"Success: {result}, Project 101 members: {PROJECTS[101]['members']}")
    print("-" * 30)

    # Owner (user 2) adds user 1 to project 101
    print("Owner (2) adds user 1 to project 101:")
    result = handle_project_membership(active_user_id=2, proj_id=101, operation='add', target_member_id=1)
    print(f"Success: {result}, Project 101 members: {PROJECTS[101]['members']}")
    print("-" * 30)

    # Owner (user 2) removes user 3 from project 101
    print("Owner (2) removes user 3 from project 101:")
    result = handle_project_membership(active_user_id=2, proj_id=101, operation='remove', target_member_id=3)
    print(f"Success: {result}, Project 101 members: {PROJECTS[101]['members']}")
    print("-" * 30)

    # Non-owner/non-admin (user 3) tries to add user 1 to project 101
    print("User 3 (non-owner/admin) tries to add user 1 to project 101:")
    result = handle_project_membership(active_user_id=3, proj_id=101, operation='add', target_member_id=1)
    print(f"Success: {result}, Project 101 members: {PROJECTS[101]['members']}")
    print("-" * 30)

    # Admin (user 1) removes user 4 (owner) from project 102
    # Note: Current logic allows admin to remove owner.
    print("Admin (1) removes user 3 (owner of 102) from project 102:")
    result = handle_project_membership(active_user_id=1, proj_id=102, operation='remove', target_member_id=3)
    print(f"Success: {result}, Project 102 members: {PROJECTS[102]['members']}")
    print("-" * 30)

    # Owner (user 4) tries to remove themselves from project 103
    print("Owner (4) tries to remove self from project 103:")
    result = handle_project_membership(active_user_id=4, proj_id=103, operation='remove', target_member_id=4)
    print(f"Success: {result}, Project 103 members: {PROJECTS[103]['members']}")
    print("-" * 30)
    
    # Admin (user 1) removes owner (user 4) from project 103
    print("Admin (1) removes owner (4) from project 103:")
    # First, let's add another member so the project isn't empty for this test, or it's fine if it becomes empty
    PROJECTS[103]['members'].append(2) # Admin adds user 2 first
    print(f"Project 103 members before admin removal: {PROJECTS[103]['members']}")
    result = handle_project_membership(active_user_id=1, proj_id=103, operation='remove', target_member_id=4)
    print(f"Success: {result}, Project 103 members: {PROJECTS[103]['members']}")
    print("-" * 30)

    # Invalid operation
    print("Admin (1) tries invalid operation 'modify' on project 101:")
    result = handle_project_membership(active_user_id=1, proj_id=101, operation='modify', target_member_id=2)
    print(f"Success: {result}, Project 101 members: {PROJECTS[101]['members']}")
    print("-" * 30)

    # Non-existent project
    print("Admin (1) tries to add user 2 to non-existent project 999:")
    result = handle_project_membership(active_user_id=1, proj_id=999, operation='add', target_member_id=2)
    print(f"Success: {result}")
    print("-" * 30)

    # Non-existent active user
    print("Non-existent user (99) tries to add user 2 to project 101:")
    result = handle_project_membership(active_user_id=99, proj_id=101, operation='add', target_member_id=2)
    print(f"Success: {result}")
    print("-" * 30)

    # Non-existent target user
    print("Admin (1) tries to add non-existent user (99) to project 101:")
    result = handle_project_membership(active_user_id=1, proj_id=101, operation='add', target_member_id=99)
    print(f"Success: {result}, Project 101 members: {PROJECTS[101]['members']}")
    print("-" * 30)

    # Adding an already existing member (should return True, no change)
    print("Owner (2) tries to add existing member (user 1) to project 101 again:")
    result = handle_project_membership(active_user_id=2, proj_id=101, operation='add', target_member_id=1)
    print(f"Success: {result}, Project 101 members: {PROJECTS[101]['members']}")
    print("-" * 30)

    # Removing a non-existent member (should return False)
    print("Owner (2) tries to remove non-member (user 3, already removed) from project 101:")
    result = handle_project_membership(active_user_id=2, proj_id=101, operation='remove', target_member_id=3)
    print(f"Success: {result}, Project 101 members: {PROJECTS[101]['members']}")
    print("-" * 30)
