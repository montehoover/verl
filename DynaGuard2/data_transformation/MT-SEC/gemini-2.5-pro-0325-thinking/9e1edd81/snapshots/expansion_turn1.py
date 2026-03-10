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

if __name__ == '__main__':
    # Example usage:
    role, owner = get_user_role_and_project_owner(1, 101)
    print(f"User 1, Project 101: Role = {role}, Owner ID = {owner}")

    role, owner = get_user_role_and_project_owner(3, 102)
    print(f"User 3, Project 102: Role = {role}, Owner ID = {owner}")

    role, owner = get_user_role_and_project_owner(5, 101) # Non-existent user
    print(f"User 5, Project 101: Role = {role}, Owner ID = {owner}")

    role, owner = get_user_role_and_project_owner(2, 200) # Non-existent project
    print(f"User 2, Project 200: Role = {role}, Owner ID = {owner}")
