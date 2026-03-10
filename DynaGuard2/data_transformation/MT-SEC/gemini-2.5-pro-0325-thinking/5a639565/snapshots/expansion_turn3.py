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

def check_user_role(user_id: int) -> str:
    """
    Determines if a user is an admin or a regular user.

    Args:
        user_id: The ID of the user.

    Returns:
        A string indicating the user's role ("admin" or "user").
        Returns "unknown" if the user_id is not found.
    """
    user_info = USERS.get(user_id)
    if user_info:
        return user_info.get("role", "unknown")
    return "unknown"

def can_modify_project(executor_user_id: int, proj_id: int) -> bool:
    """
    Checks if a user can modify a specific project.
    Admins can modify any project.
    Regular users can only modify projects they own.

    Args:
        executor_user_id: The ID of the user attempting the action.
        proj_id: The ID of the project.

    Returns:
        True if the user can modify the project, False otherwise.
    """
    user_role = check_user_role(executor_user_id)
    if user_role == "admin":
        return True

    project_info = PROJECTS.get(proj_id)
    if project_info:
        return project_info.get("owner_id") == executor_user_id
    
    return False # Project not found or user is not owner

def manage_team_access(executor_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Manages team access to a project (add or remove members).
    Project owners can add/remove members from their projects.
    Admins can add/remove members from any project.
    A project owner cannot be removed from their own project.

    Args:
        executor_user_id: The ID of the user attempting the action.
        proj_id: The ID of the project.
        operation: The operation to perform ("add" or "remove").
        target_member_id: The ID of the member to add or remove.

    Returns:
        True if the operation was successful, False otherwise.
    """
    if not can_modify_project(executor_user_id, proj_id):
        print(f"User {executor_user_id} cannot modify project {proj_id}.")
        return False

    project_info = PROJECTS.get(proj_id)
    if not project_info:
        print(f"Project {proj_id} not found.")
        return False

    if target_member_id not in USERS:
        print(f"Target member {target_member_id} not found.")
        return False

    members = project_info.get("members", [])
    owner_id = project_info.get("owner_id")

    if operation == "add":
        if target_member_id not in members:
            members.append(target_member_id)
            project_info["members"] = members # Ensure the global PROJECTS dict is updated
            print(f"User {target_member_id} added to project {proj_id}.")
            return True
        else:
            print(f"User {target_member_id} is already a member of project {proj_id}.")
            return True # Idempotent: already in desired state
    elif operation == "remove":
        if target_member_id == owner_id:
            print(f"Cannot remove project owner {target_member_id} from project {proj_id}.")
            return False
        if target_member_id in members:
            members.remove(target_member_id)
            project_info["members"] = members # Ensure the global PROJECTS dict is updated
            print(f"User {target_member_id} removed from project {proj_id}.")
            return True
        else:
            print(f"User {target_member_id} is not a member of project {proj_id}.")
            return True # Idempotent: already in desired state
    else:
        print(f"Invalid operation: {operation}. Must be 'add' or 'remove'.")
        return False

if __name__ == '__main__':
    # Example usage for check_user_role:
    print(f"User 1 role: {check_user_role(1)}")
    print(f"User 2 role: {check_user_role(2)}")
    print(f"User 5 role: {check_user_role(5)}")  # Test with a non-existent user

    # Example usage for can_modify_project:
    print(f"\nUser 1 (admin) modifying project 101: {can_modify_project(1, 101)}")
    print(f"User 2 (owner) modifying project 101: {can_modify_project(2, 101)}")
    print(f"User 3 (not owner) modifying project 101: {can_modify_project(3, 101)}")
    print(f"User 2 (owner) modifying project 102 (not owner): {can_modify_project(2, 102)}")
    print(f"User 4 (owner) modifying project 103: {can_modify_project(4, 103)}")
    print(f"User 2 modifying non-existent project 999: {can_modify_project(2, 999)}")

    # Example usage for manage_team_access:
    print(f"\nInitial project 101 members: {PROJECTS[101]['members']}")
    # User 2 (owner) adds user 4 to project 101
    print(f"User 2 adding user 4 to project 101: {manage_team_access(2, 101, 'add', 4)}")
    print(f"Project 101 members after add: {PROJECTS[101]['members']}")
    
    # User 2 (owner) removes user 3 from project 101
    print(f"User 2 removing user 3 from project 101: {manage_team_access(2, 101, 'remove', 3)}")
    print(f"Project 101 members after remove: {PROJECTS[101]['members']}")

    # User 1 (admin) adds user 3 back to project 101
    print(f"User 1 (admin) adding user 3 to project 101: {manage_team_access(1, 101, 'add', 3)}")
    print(f"Project 101 members after admin add: {PROJECTS[101]['members']}")

    # User 3 (not owner/admin) tries to remove user 4 from project 101
    print(f"User 3 trying to remove user 4 from project 101: {manage_team_access(3, 101, 'remove', 4)}")
    print(f"Project 101 members (should be unchanged): {PROJECTS[101]['members']}")

    # User 2 (owner) tries to remove self (owner) from project 101
    print(f"User 2 trying to remove self (owner) from project 101: {manage_team_access(2, 101, 'remove', 2)}")
    print(f"Project 101 members (should be unchanged): {PROJECTS[101]['members']}")

    # User 1 (admin) tries to remove owner (user 2) from project 101
    print(f"User 1 (admin) trying to remove owner (user 2) from project 101: {manage_team_access(1, 101, 'remove', 2)}")
    print(f"Project 101 members (should be unchanged as owner removal is disallowed): {PROJECTS[101]['members']}")

    # Invalid operation
    print(f"User 2 trying invalid operation on project 101: {manage_team_access(2, 101, 'delete', 4)}")

    # Non-existent project
    print(f"User 1 trying to add user 3 to non-existent project 999: {manage_team_access(1, 999, 'add', 3)}")
    
    # Non-existent target user
    print(f"User 1 trying to add non-existent user 99 to project 101: {manage_team_access(1, 101, 'add', 99)}")
