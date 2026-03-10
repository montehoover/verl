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
    Determine a user's role.

    Args:
        user_id: The integer user ID.

    Returns:
        A string representing the user's role, e.g., "admin" or "user".

    Raises:
        TypeError: If user_id is not an integer.
        ValueError: If the user_id does not exist in the USERS mapping.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    try:
        return USERS[user_id]["role"]
    except KeyError:
        raise ValueError(f"Unknown user_id: {user_id}")


def can_modify_project(executing_user_id: int, prjct_id: int) -> bool:
    """
    Determine whether a user can modify a project.

    Rules:
    - Admins can modify any project.
    - The project owner can modify their project.

    Args:
        executing_user_id: The integer user ID performing the action.
        prjct_id: The integer project ID.

    Returns:
        True if the user can modify the project, otherwise False.

    Raises:
        TypeError: If inputs are not integers.
        ValueError: If the project_id or user_id are unknown.
    """
    if not isinstance(executing_user_id, int):
        raise TypeError("executing_user_id must be an integer")
    if not isinstance(prjct_id, int):
        raise TypeError("prjct_id must be an integer")

    try:
        project = PROJECTS[prjct_id]
    except KeyError:
        raise ValueError(f"Unknown project_id: {prjct_id}")

    role = check_user_role(executing_user_id)
    if role == "admin":
        return True

    return project["owner_id"] == executing_user_id
