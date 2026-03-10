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

def control_project_participants(acting_user_id: int, prj_id: int, action_type: str, target_id: int) -> bool:
    """
    Manages adding or removing members from a project.
    Requires 'add_member' or 'remove_member' action type.
    Users must have permission to manage the project.
    """
    if not can_perform_action(acting_user_id, prj_id, action_type):
        print(f"Debug: User {acting_user_id} does not have permission for action '{action_type}' on project {prj_id}.")
        return False

    project = PROJECTS.get(prj_id)
    if not project:
        # Project not found, though can_perform_action should catch this via has_permission
        print(f"Debug: Project {prj_id} not found in control_project_participants.")
        return False

    target_user_exists = target_id in USERS
    if not target_user_exists:
        print(f"Debug: Target user {target_id} does not exist.")
        return False

    if action_type == "add_member":
        if target_id not in project["members"]:
            project["members"].append(target_id)
            print(f"Debug: User {target_id} added to project {prj_id} by user {acting_user_id}.")
            return True
        else:
            print(f"Debug: User {target_id} is already a member of project {prj_id}.")
            return False  # Or True, if idempotency is desired and no change is success
    elif action_type == "remove_member":
        if target_id == project["owner_id"]:
            print(f"Debug: Cannot remove project owner ({target_id}) from project {prj_id}.")
            return False # Owners cannot be removed this way
        if target_id in project["members"]:
            project["members"].remove(target_id)
            print(f"Debug: User {target_id} removed from project {prj_id} by user {acting_user_id}.")
            return True
        else:
            print(f"Debug: User {target_id} is not a member of project {prj_id}, cannot remove.")
            return False
    else:
        print(f"Debug: Invalid action_type '{action_type}' for control_project_participants.")
        return False # Invalid action

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

    print("\n-- Control Project Participants Tests --")
    # Initial state for reference
    print(f"Initial P101 members: {PROJECTS[101]['members']}")
    print(f"Initial P102 members: {PROJECTS[102]['members']}")

    # Admin adds user 4 to project 101
    print(f"Admin (1) adds user 4 to P101: {control_project_participants(1, 101, 'add_member', 4)}")  # Expected: True
    print(f"P101 members after admin add: {PROJECTS[101]['members']}") # Expected: [2, 3, 4]

    # Owner (user 2) removes user 3 from project 101
    print(f"User 2 (owner) removes user 3 from P101: {control_project_participants(2, 101, 'remove_member', 3)}")  # Expected: True
    print(f"P101 members after owner remove: {PROJECTS[101]['members']}") # Expected: [2, 4]

    # User 4 (not owner/admin) tries to add user 3 to project 101 (should fail permission)
    print(f"User 4 (non-owner) tries to add user 3 to P101: {control_project_participants(4, 101, 'add_member', 3)}")  # Expected: False
    print(f"P101 members after failed add: {PROJECTS[101]['members']}") # Expected: [2, 4]

    # Owner (user 3) tries to remove self (owner) from project 102 (should fail, owner cannot remove self)
    print(f"User 3 (owner) tries to remove self from P102: {control_project_participants(3, 102, 'remove_member', 3)}")  # Expected: False
    print(f"P102 members after failed self-remove: {PROJECTS[102]['members']}") # Expected: [3, 4]

    # Admin (user 1) tries to remove owner (user 3) from project 102 (should fail, owner cannot remove self)
    print(f"Admin (1) tries to remove owner (3) from P102: {control_project_participants(1, 102, 'remove_member', 3)}")  # Expected: False
    print(f"P102 members after admin failed owner-remove: {PROJECTS[102]['members']}") # Expected: [3, 4]

    # Owner (user 2) tries to add non-existent user 5 to project 101
    print(f"User 2 (owner) tries to add non-existent user 5 to P101: {control_project_participants(2, 101, 'add_member', 5)}")  # Expected: False
    print(f"P101 members after failed add non-existent: {PROJECTS[101]['members']}") # Expected: [2, 4]

    # Owner (user 2) tries to remove user 3 (already removed) from project 101
    print(f"User 2 (owner) tries to remove user 3 (not member) from P101: {control_project_participants(2, 101, 'remove_member', 3)}")  # Expected: False
    print(f"P101 members after failed remove non-member: {PROJECTS[101]['members']}") # Expected: [2, 4]
    
    # Admin (user 1) adds user 2 to project 103
    print(f"Admin (1) adds user 2 to P103: {control_project_participants(1, 103, 'add_member', 2)}") # Expected: True
    print(f"P103 members after admin add: {PROJECTS[103]['members']}") # Expected: [4, 2]

    # Admin (user 1) removes user 4 (owner) from project 103 - this should fail
    print(f"Admin (1) tries to remove owner (4) from P103: {control_project_participants(1, 103, 'remove_member', 4)}") # Expected: False
    print(f"P103 members after failed owner removal: {PROJECTS[103]['members']}") # Expected: [4, 2]

    # User 2 (now member of P103, but not owner) tries to remove user 4 (owner) from P103 - this should fail (permission)
    print(f"User 2 (member) tries to remove owner (4) from P103: {control_project_participants(2, 103, 'remove_member', 4)}") # Expected: False
    print(f"P103 members after failed owner removal by member: {PROJECTS[103]['members']}") # Expected: [4, 2]
