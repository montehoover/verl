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
    Determine the role for the given user_id.

    Args:
        user_id (int): The user's ID.

    Returns:
        str: "admin" or "user".

    Raises:
        TypeError: If user_id is not an integer.
        ValueError: If the user is not found or has an invalid role.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")

    user = USERS.get(user_id)
    if user is None:
        raise ValueError(f"User with id {user_id} not found")

    role = user.get("role")
    if role not in {"admin", "user"}:
        raise ValueError(f"Invalid role for user {user_id}: {role}")

    return role


def can_manage_project(user_id: int, prj_id: int) -> bool:
    """
    Determine whether a user can manage a given project.

    A user can manage a project if:
    - The user is an admin, or
    - The user is the owner of the project.

    Args:
        user_id (int): The user's ID.
        prj_id (int): The project's ID.

    Returns:
        bool: True if the user can manage the project, False otherwise.

    Raises:
        TypeError: If user_id or prj_id are not integers.
        ValueError: If the user or project is not found.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")
    if not isinstance(prj_id, int):
        raise TypeError("prj_id must be an int")

    # Validate user and get role
    role = check_user_role(user_id)

    # Validate project
    project = PROJECTS.get(prj_id)
    if project is None:
        raise ValueError(f"Project with id {prj_id} not found")

    if role == "admin":
        return True

    if project.get("owner_id") == user_id:
        return True

    return False
