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
