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
        A string indicating the user's role ("admin", "user", or "unknown").
    """
    user_info = USERS.get(user_id)
    if user_info:
        return user_info.get("role", "unknown")
    return "unknown"

def can_manage_project(user_id: int, prj_id: int) -> bool:
    """
    Checks if a user can manage a specific project.
    Admins can manage any project.
    Project owners can manage their projects.

    Args:
        user_id: The ID of the user.
        prj_id: The ID of the project.

    Returns:
        True if the user can manage the project, False otherwise.
    """
    user_role = check_user_role(user_id)
    if user_role == "admin":
        return True

    project_info = PROJECTS.get(prj_id)
    if project_info:
        if project_info.get("owner_id") == user_id:
            return True
    
    return False

def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    """
    Manages project membership (add/remove members).
    Admins can manage any project.
    Project owners can manage their own projects.

    Args:
        active_user_id: The ID of the user performing the action.
        prj_id: The ID of the project.
        action_type: "add" or "remove".
        member_user_id: The ID of the user to be added or removed.

    Returns:
        True if the operation is successful, False otherwise.
    """
    if prj_id not in PROJECTS:
        print(f"Error: Project {prj_id} not found.")
        return False

    if member_user_id not in USERS:
        print(f"Error: User {member_user_id} not found.")
        return False

    if not can_manage_project(active_user_id, prj_id):
        print(f"Error: User {active_user_id} does not have permission to manage project {prj_id}.")
        return False

    project = PROJECTS[prj_id]
    members = project["members"]

    if action_type == "add":
        if member_user_id not in members:
            members.append(member_user_id)
            print(f"User {member_user_id} added to project {prj_id}.")
            return True
        else:
            print(f"User {member_user_id} is already a member of project {prj_id}.")
            return False
    elif action_type == "remove":
        if member_user_id == project["owner_id"]:
            print(f"Error: Cannot remove project owner (User {member_user_id}) from project {prj_id}.")
            return False
        if member_user_id in members:
            members.remove(member_user_id)
            print(f"User {member_user_id} removed from project {prj_id}.")
            return True
        else:
            print(f"User {member_user_id} is not a member of project {prj_id}.")
            return False
    else:
        print(f"Error: Invalid action type '{action_type}'. Must be 'add' or 'remove'.")
        return False

if __name__ == '__main__':
    # Example usage for check_user_role:
    print(f"User 1 role: {check_user_role(1)}")
    print(f"User 2 role: {check_user_role(2)}")
    print(f"User 5 role: {check_user_role(5)}")

    # Example usage for can_manage_project:
    print(f"\nUser 1 (admin) can manage project 101: {can_manage_project(1, 101)}")
    print(f"User 2 (owner) can manage project 101: {can_manage_project(2, 101)}")
    print(f"User 2 (not owner) can manage project 102: {can_manage_project(2, 102)}")
    print(f"User 3 (owner) can manage project 102: {can_manage_project(3, 102)}")
    print(f"User 4 (not admin, not owner) can manage project 101: {can_manage_project(4, 101)}")
    print(f"User 2 can manage non-existent project 999: {can_manage_project(2, 999)}")

    # Example usage for project_access_control:
    print("\n--- Project Access Control ---")
    # Admin (User 1) adds User 4 to Project 101
    print(f"Initial members of Project 101: {PROJECTS[101]['members']}")
    project_access_control(1, 101, "add", 4)
    print(f"Members of Project 101 after admin adds User 4: {PROJECTS[101]['members']}")

    # Owner (User 2) removes User 3 from Project 101
    project_access_control(2, 101, "remove", 3)
    print(f"Members of Project 101 after owner (User 2) removes User 3: {PROJECTS[101]['members']}")

    # Owner (User 2) tries to remove self (owner) from Project 101 - Should fail
    project_access_control(2, 101, "remove", 2)
    print(f"Members of Project 101 after owner (User 2) tries to remove self: {PROJECTS[101]['members']}")

    # Non-owner (User 3) tries to add User 1 to Project 101 - Should fail
    project_access_control(3, 101, "add", 1)
    print(f"Members of Project 101 after User 3 tries to add User 1: {PROJECTS[101]['members']}")
    
    # Admin (User 1) removes User 4 from Project 102 (User 4 is not owner)
    print(f"Initial members of Project 102: {PROJECTS[102]['members']}")
    project_access_control(1, 102, "remove", 4)
    print(f"Members of Project 102 after admin removes User 4: {PROJECTS[102]['members']}")

    # Add User 4 back for further tests
    PROJECTS[102]['members'].append(4) 

    # Owner (User 3) tries to remove non-member (User 1) from Project 102 - Should fail (gracefully)
    project_access_control(3, 102, "remove", 1)
    print(f"Members of Project 102: {PROJECTS[102]['members']}")

    # Invalid action
    project_access_control(1, 101, "modify", 4)

    # Non-existent project
    project_access_control(1, 999, "add", 4)

    # Non-existent member to add
    project_access_control(1, 101, "add", 99)
