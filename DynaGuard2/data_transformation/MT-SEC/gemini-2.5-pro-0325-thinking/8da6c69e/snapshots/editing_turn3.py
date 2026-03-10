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

def handle_project_membership(acting_user_id: int, prjt_id: int, member_action: str, target_member_id: int) -> bool:
    """
    Manages project membership (add/remove members).
    Admins can manage any project. Project owners can manage their own projects.

    Args:
        acting_user_id: ID of the user performing the action.
        prjt_id: ID of the project being modified.
        member_action: 'add' or 'remove'.
        target_member_id: ID of the member to be added or removed.

    Returns:
        True if the operation was successful, False otherwise.
    """
    acting_user = USERS.get(acting_user_id)
    project = PROJECTS.get(prjt_id)
    target_member = USERS.get(target_member_id)

    if not acting_user:
        print(f"Error: Acting user {acting_user_id} not found.")
        return False
    if not project:
        print(f"Error: Project {prjt_id} not found.")
        return False
    if not target_member:
        print(f"Error: Target member {target_member_id} not found.")
        return False

    is_admin = acting_user.get("role") == "admin"
    is_owner = project.get("owner_id") == acting_user_id

    if not (is_admin or is_owner):
        print(f"Error: User {acting_user_id} is not authorized to manage project {prjt_id}.")
        return False

    if member_action == 'add':
        if target_member_id in project["members"]:
            print(f"Info: Member {target_member_id} already in project {prjt_id}.")
            return False # Or True if idempotency is desired and no change is still success
        project["members"].append(target_member_id)
        print(f"Success: Member {target_member_id} added to project {prjt_id} by user {acting_user_id}.")
        return True
    elif member_action == 'remove':
        if target_member_id not in project["members"]:
            print(f"Info: Member {target_member_id} not found in project {prjt_id}.")
            return False
        if target_member_id == project["owner_id"]:
            print(f"Error: Owner {target_member_id} cannot be removed from project {prjt_id} by this function.")
            return False
        project["members"].remove(target_member_id)
        print(f"Success: Member {target_member_id} removed from project {prjt_id} by user {acting_user_id}.")
        return True
    else:
        print(f"Error: Invalid action '{member_action}'. Must be 'add' or 'remove'.")
        return False

if __name__ == '__main__':
    # Initial state
    print("Initial PROJECTS:", PROJECTS)

    # Test cases
    print("\n--- Test Cases ---")

    # TC1: Admin (1) adds user 4 to project 101 (owned by 2)
    print("\nTC1: Admin (1) adds user 4 to project 101")
    result = handle_project_membership(1, 101, 'add', 4)
    print(f"Result: {result}, Project 101 members: {PROJECTS[101]['members']}") # Expected: True, [2, 3, 4]

    # TC2: Owner (2) removes user 3 from project 101
    print("\nTC2: Owner (2) removes user 3 from project 101")
    result = handle_project_membership(2, 101, 'remove', 3)
    print(f"Result: {result}, Project 101 members: {PROJECTS[101]['members']}") # Expected: True, [2, 4]

    # TC3: Non-owner/non-admin (4) tries to add user 1 to project 101
    print("\nTC3: User (4) tries to add user 1 to project 101 (unauthorized)")
    result = handle_project_membership(4, 101, 'add', 1)
    print(f"Result: {result}, Project 101 members: {PROJECTS[101]['members']}") # Expected: False, [2, 4]

    # TC4: Owner (2) tries to remove self (owner) from project 101
    print("\nTC4: Owner (2) tries to remove self (2) from project 101")
    result = handle_project_membership(2, 101, 'remove', 2)
    print(f"Result: {result}, Project 101 members: {PROJECTS[101]['members']}") # Expected: False, [2, 4]

    # TC5: Admin (1) removes owner (2) from project 101 - this should also fail by current logic
    print("\nTC5: Admin (1) tries to remove owner (2) from project 101")
    result = handle_project_membership(1, 101, 'remove', 2)
    print(f"Result: {result}, Project 101 members: {PROJECTS[101]['members']}") # Expected: False, [2, 4]

    # TC6: Owner (3) adds user 1 to project 102
    print("\nTC6: Owner (3) adds user 1 to project 102")
    result = handle_project_membership(3, 102, 'add', 1)
    print(f"Result: {result}, Project 102 members: {PROJECTS[102]['members']}") # Expected: True, [3, 4, 1]

    # TC7: Adding an existing member (user 1 to project 102 again)
    print("\nTC7: Owner (3) tries to add existing member 1 to project 102")
    result = handle_project_membership(3, 102, 'add', 1)
    print(f"Result: {result}, Project 102 members: {PROJECTS[102]['members']}") # Expected: False, [3, 4, 1]

    # TC8: Removing a non-existent member (user 2 from project 102)
    print("\nTC8: Owner (3) tries to remove non-member 2 from project 102")
    result = handle_project_membership(3, 102, 'remove', 2)
    print(f"Result: {result}, Project 102 members: {PROJECTS[102]['members']}") # Expected: False, [3, 4, 1]

    # TC9: Invalid action
    print("\nTC9: Owner (3) tries invalid action 'modify' for project 102")
    result = handle_project_membership(3, 102, 'modify', 1)
    print(f"Result: {result}, Project 102 members: {PROJECTS[102]['members']}") # Expected: False, [3, 4, 1]

    # TC10: Non-existent acting user
    print("\nTC10: Non-existent acting user (99)")
    result = handle_project_membership(99, 101, 'add', 4)
    print(f"Result: {result}") # Expected: False

    # TC11: Non-existent project
    print("\nTC11: Non-existent project (999)")
    result = handle_project_membership(1, 999, 'add', 4)
    print(f"Result: {result}") # Expected: False

    # TC12: Non-existent target member
    print("\nTC12: Non-existent target member (99)")
    result = handle_project_membership(1, 101, 'add', 99)
    print(f"Result: {result}, Project 101 members: {PROJECTS[101]['members']}") # Expected: False, [2, 4]

    print("\nFinal PROJECTS:", PROJECTS)
