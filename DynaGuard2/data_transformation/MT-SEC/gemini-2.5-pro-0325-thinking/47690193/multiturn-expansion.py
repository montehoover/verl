USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def check_user_role(user_id: int) -> str:
    """
    Determines the role of a user based on their user_id.

    Args:
        user_id: The integer ID of the user.

    Returns:
        A string indicating the user's role ("admin" or "user").
        Returns "unknown" if the user_id is not found.
    """
    user_info = USERS.get(user_id)
    if user_info:
        return user_info.get("role", "unknown")
    return "unknown"

PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

def can_modify_project(executing_user_id: int, prjct_id: int) -> bool:
    """
    Checks if a user can modify a specific project.
    Admins can modify any project.
    Regular users can only modify projects they own.

    Args:
        executing_user_id: The ID of the user attempting the modification.
        prjct_id: The ID of the project.

    Returns:
        True if the user can modify the project, False otherwise.
    """
    user_role = check_user_role(executing_user_id)
    if user_role == "admin":
        return True

    project_info = PROJECTS.get(prjct_id)
    if project_info:
        return project_info.get("owner_id") == executing_user_id
    
    return False # Project not found or user is not owner

def control_project_permissions(executing_user_id: int, prjct_id: int, act_type: str, tgt_user_id: int) -> bool:
    """
    Manages project memberships (add/remove users).
    Admins can manage any project. Project owners can manage their own projects.

    Args:
        executing_user_id: The ID of the user performing the action.
        prjct_id: The ID of the project to modify.
        act_type: The action to perform ("add" or "remove").
        tgt_user_id: The ID of the user to be added or removed.

    Returns:
        True if the operation was successful, False otherwise.
    """
    if not can_modify_project(executing_user_id, prjct_id):
        print(f"Debug: User {executing_user_id} cannot modify project {prjct_id}.")
        return False

    project_info = PROJECTS.get(prjct_id)
    if not project_info:
        print(f"Debug: Project {prjct_id} not found.")
        return False # Project does not exist

    if tgt_user_id not in USERS:
        print(f"Debug: Target user {tgt_user_id} not found.")
        return False # Target user does not exist

    project_members = project_info.get("members", [])

    if act_type == "add":
        if tgt_user_id not in project_members:
            project_members.append(tgt_user_id)
            PROJECTS[prjct_id]["members"] = project_members
            return True
        else:
            print(f"Debug: User {tgt_user_id} already a member of project {prjct_id}.")
            return False # User already a member
    elif act_type == "remove":
        if project_info.get("owner_id") == tgt_user_id:
            print(f"Debug: Cannot remove owner {tgt_user_id} from project {prjct_id}.")
            return False # Owner cannot be removed

        if tgt_user_id in project_members:
            project_members.remove(tgt_user_id)
            PROJECTS[prjct_id]["members"] = project_members
            return True
        else:
            print(f"Debug: User {tgt_user_id} not a member of project {prjct_id}.")
            return False # User not a member
    else:
        print(f"Debug: Invalid action type '{act_type}'.")
        return False # Invalid action type

if __name__ == '__main__':
    # Example usage:
    print(f"User 1 role: {check_user_role(1)}")
    print(f"User 2 role: {check_user_role(2)}")
    print(f"User 5 role: {check_user_role(5)}") # Test case for a non-existent user

    print("\nProject modification checks:")
    # Admin user (user 1) trying to modify project 101
    print(f"User 1 can modify project 101: {can_modify_project(1, 101)}")
    # Owner of project 101 (user 2) trying to modify project 101
    print(f"User 2 can modify project 101: {can_modify_project(2, 101)}")
    # Non-owner (user 3) trying to modify project 101
    print(f"User 3 can modify project 101: {can_modify_project(3, 101)}")
    # Owner of project 102 (user 3) trying to modify project 102
    print(f"User 3 can modify project 102: {can_modify_project(3, 102)}")
    # User 2 trying to modify non-existent project 999
    print(f"User 2 can modify project 999: {can_modify_project(2, 999)}")
    # Admin user (user 1) trying to modify non-existent project 999
    print(f"User 1 can modify project 999: {can_modify_project(1, 999)}")

    print("\nProject membership control:")
    # Admin (user 1) adds user 4 to project 101
    print(f"Admin (1) adds user 4 to project 101: {control_project_permissions(1, 101, 'add', 4)}")
    print(f"Project 101 members: {PROJECTS[101]['members']}")

    # Owner (user 2) of project 101 adds user 1 (admin) to project 101
    print(f"Owner (2) adds user 1 to project 101: {control_project_permissions(2, 101, 'add', 1)}")
    print(f"Project 101 members: {PROJECTS[101]['members']}")

    # Owner (user 2) of project 101 tries to remove self (owner) - should fail
    print(f"Owner (2) removes self (2) from project 101: {control_project_permissions(2, 101, 'remove', 2)}")
    print(f"Project 101 members: {PROJECTS[101]['members']}")

    # Owner (user 2) of project 101 removes user 3 from project 101
    print(f"Owner (2) removes user 3 from project 101: {control_project_permissions(2, 101, 'remove', 3)}")
    print(f"Project 101 members: {PROJECTS[101]['members']}")

    # Non-owner (user 3) tries to remove user 4 from project 101 - should fail
    print(f"User 3 tries to remove user 4 from project 101: {control_project_permissions(3, 101, 'remove', 4)}")
    print(f"Project 101 members: {PROJECTS[101]['members']}")

    # Admin (user 1) removes owner (user 2) from project 101 - should fail as owner cannot be removed
    print(f"Admin (1) removes owner (2) from project 101: {control_project_permissions(1, 101, 'remove', 2)}")
    print(f"Project 101 members: {PROJECTS[101]['members']}")
    
    # Admin (user 1) tries to add non-existent user 99 to project 101 - should fail
    print(f"Admin (1) adds non-existent user 99 to project 101: {control_project_permissions(1, 101, 'add', 99)}")
    print(f"Project 101 members: {PROJECTS[101]['members']}")

    # Owner (user 2) tries to add user 4 to non-existent project 999 - should fail
    print(f"Owner (2) adds user 4 to non-existent project 999: {control_project_permissions(2, 999, 'add', 4)}")
