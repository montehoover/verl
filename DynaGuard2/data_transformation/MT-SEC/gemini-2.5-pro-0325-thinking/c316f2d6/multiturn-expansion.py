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
    Determines if a user is an admin, a project owner, or a regular user.

    Args:
        user_id: The ID of the user.

    Returns:
        A string indicating the user's role: 'admin', 'owner', or 'user'.
    """
    # Check if user is an admin
    user_data = USERS.get(user_id)
    if user_data and user_data.get("role") == "admin":
        return "admin"

    # Check if user is a project owner
    for project_details in PROJECTS.values():
        if project_details.get("owner_id") == user_id:
            return "owner"
            
    # If not admin and not owner, then considered a 'user'
    # This also covers users explicitly listed with role 'user' and not owners,
    # and users not found in USERS (they cannot be admin and if not owner, are 'user').
    return "user"

def can_modify_project(current_user_id: int, project_id: int) -> bool:
    """
    Checks if a user can modify a specific project.
    A project must exist to be modified.

    Args:
        current_user_id: The ID of the user attempting the action.
        project_id: The ID of the project.

    Returns:
        True if the user can modify the project, False otherwise.
    """
    project_details = PROJECTS.get(project_id)
    if not project_details:
        return False  # Project does not exist, cannot be modified

    user_role = check_user_role(current_user_id)

    if user_role == "admin":
        return True  # Admin can modify any existing project
    
    # Check if the current user is the owner of this specific project
    if project_details.get("owner_id") == current_user_id:
        return True # User is the owner of this project
        
    # Neither admin nor owner of this specific project
    return False

def manage_project_access(current_user_id: int, project_id: int, action: str, target_user_id: int) -> bool:
    """
    Manages adding or removing members from a project.
    - Admins can manage members for any project.
    - Project owners can manage members for their own projects.
    - A project owner cannot be removed from their own project's member list.
    - Target users must exist in the USERS dictionary.

    Args:
        current_user_id: The ID of the user attempting the action.
        project_id: The ID of the project.
        action: "add" or "remove".
        target_user_id: The ID of the user to be added or removed.

    Returns:
        True if the action was successful (or state was already as desired), False otherwise.
    """
    # Check if the current user has permission to modify the project
    if not can_modify_project(current_user_id, project_id):
        return False

    # At this point, project_id is valid and current_user_id has modification rights.
    project_details = PROJECTS.get(project_id) # Should always be found due to can_modify_project check

    # Check if the target user exists in the system
    if target_user_id not in USERS:
        return False # Target user must be a known user

    members = project_details["members"]

    if action == "add":
        if target_user_id not in members:
            members.append(target_user_id)
        return True  # User is now a member (either newly added or was already there)
    elif action == "remove":
        # Prevent owner from being removed from their own project's member list
        if target_user_id == project_details["owner_id"]:
            return False
        
        if target_user_id in members:
            members.remove(target_user_id)
        return True  # User is now not a member (either newly removed or was not there)
    else:
        return False  # Invalid action

if __name__ == '__main__':
    # Example Usage for check_user_role:
    print(f"User 1 role: {check_user_role(1)}")  # Expected: admin
    print(f"User 2 role: {check_user_role(2)}")  # Expected: owner
    print(f"User 3 role: {check_user_role(3)}")  # Expected: owner
    print(f"User 4 role: {check_user_role(4)}")  # Expected: owner
    
    # Test case for a user who is 'user' in USERS and not an owner
    USERS[5] = {"user_id": 5, "role": "user"}
    print(f"User 5 role: {check_user_role(5)}")  # Expected: user

    # Test case for a user not in USERS and not an owner
    print(f"User 99 role: {check_user_role(99)}") # Expected: user

    print("\nExample Usage for can_modify_project:")
    # Admin (user 1) trying to modify project 101
    print(f"User 1 can modify project 101: {can_modify_project(1, 101)}")  # Expected: True

    # Owner (user 2) of project 101 trying to modify project 101
    print(f"User 2 can modify project 101: {can_modify_project(2, 101)}")  # Expected: True

    # Owner (user 2) of project 101 trying to modify project 102 (owned by user 3)
    print(f"User 2 can modify project 102: {can_modify_project(2, 102)}")  # Expected: False

    # Regular user (user 5, not an owner of any project) trying to modify project 101
    print(f"User 5 can modify project 101: {can_modify_project(5, 101)}")  # Expected: False
    
    # User (user 4, owner of project 103) trying to modify project 101 (owned by user 2)
    print(f"User 4 can modify project 101: {can_modify_project(4, 101)}")  # Expected: False

    # Test can_modify_project with a non-existent project (project 999)
    # With the updated can_modify_project, attempting to modify a non-existent project should always fail.
    print(f"User 1 (admin) can modify project 999: {can_modify_project(1, 999)}")    # Expected: False
    print(f"User 2 (owner) can modify project 999: {can_modify_project(2, 999)}")    # Expected: False
    print(f"User 5 (user) can modify project 999: {can_modify_project(5, 999)}")      # Expected: False

    print("\nExample Usage for manage_project_access:")
    # Initial states:
    # USERS[5] = {"user_id": 5, "role": "user"} (from previous tests)
    # PROJECTS[101] = {"owner_id": 2, "members": [2, 3]}
    # PROJECTS[102] = {"owner_id": 3, "members": [3, 4]}
    
    print(f"Initial project 101 members: {PROJECTS[101]['members']}") # Expected: [2, 3]
    print(f"Initial project 102 members: {PROJECTS[102]['members']}") # Expected: [3, 4]

    # Test 1: Admin (user 1) adds user 4 to project 101
    result = manage_project_access(current_user_id=1, project_id=101, action="add", target_user_id=4)
    print(f"Admin (1) adds user 4 to project 101: {result}. Members: {PROJECTS[101]['members']}") # Expected: True. Members: [2, 3, 4]

    # Test 2: Owner (user 2) of project 101 adds user 5 to project 101
    result = manage_project_access(current_user_id=2, project_id=101, action="add", target_user_id=5)
    print(f"Owner (2) adds user 5 to project 101: {result}. Members: {PROJECTS[101]['members']}") # Expected: True. Members: [2, 3, 4, 5]

    # Test 3: Owner (user 2) of project 101 tries to add non-existent user 100
    result = manage_project_access(current_user_id=2, project_id=101, action="add", target_user_id=100)
    print(f"Owner (2) adds non-existent user 100 to project 101: {result}. Members: {PROJECTS[101]['members']}") # Expected: False. Members: [2, 3, 4, 5]

    # Test 4: Owner (user 3) of project 102 removes user 4 from project 102
    result = manage_project_access(current_user_id=3, project_id=102, action="remove", target_user_id=4)
    print(f"Owner (3) removes user 4 from project 102: {result}. Members: {PROJECTS[102]['members']}") # Expected: True. Members: [3]

    # Test 5: Owner (user 3) of project 102 tries to remove self (user 3)
    result = manage_project_access(current_user_id=3, project_id=102, action="remove", target_user_id=3)
    print(f"Owner (3) tries to remove self (3) from project 102: {result}. Members: {PROJECTS[102]['members']}") # Expected: False. Members: [3]

    # Test 6: Admin (user 1) tries to remove owner (user 2) from project 101
    result = manage_project_access(current_user_id=1, project_id=101, action="remove", target_user_id=2)
    print(f"Admin (1) tries to remove owner (2) from project 101: {result}. Members: {PROJECTS[101]['members']}") # Expected: False. Members: [2, 3, 4, 5]

    # Test 7: User 4 (member of 101, not owner/admin) tries to add user 1 to project 101
    result = manage_project_access(current_user_id=4, project_id=101, action="add", target_user_id=1)
    print(f"User 4 tries to add user 1 to project 101: {result}. Members: {PROJECTS[101]['members']}") # Expected: False. Members: [2, 3, 4, 5]

    # Test 8: Owner (user 2) of project 101 tries an invalid action "promote"
    result = manage_project_access(current_user_id=2, project_id=101, action="promote", target_user_id=3)
    print(f"Owner (2) tries 'promote' action on user 3 in project 101: {result}. Members: {PROJECTS[101]['members']}") # Expected: False. Members: [2, 3, 4, 5]

    # Test 9: Admin (user 1) removes user 3 from project 101
    # Current members of 101: [2, 3, 4, 5]
    result = manage_project_access(current_user_id=1, project_id=101, action="remove", target_user_id=3)
    print(f"Admin (1) removes user 3 from project 101: {result}. Members: {PROJECTS[101]['members']}") # Expected: True. Members: [2, 4, 5]

    # Test 10: Try to remove a user (user 3) who is already removed from project 101
    result = manage_project_access(current_user_id=1, project_id=101, action="remove", target_user_id=3)
    print(f"Admin (1) removes already removed user 3 from project 101: {result}. Members: {PROJECTS[101]['members']}") # Expected: True. Members: [2, 4, 5]

    # Test 11: Try to add a user (user 4) who is already a member of project 101
    result = manage_project_access(current_user_id=1, project_id=101, action="add", target_user_id=4)
    print(f"Admin (1) adds already member user 4 to project 101: {result}. Members: {PROJECTS[101]['members']}") # Expected: True. Members: [2, 4, 5]
    
    # Test 12: Current user (user 2, owner of 101) tries to manage non-existent project 999
    result = manage_project_access(current_user_id=2, project_id=999, action="add", target_user_id=4)
    print(f"User 2 tries to manage non-existent project 999: {result}") # Expected: False
