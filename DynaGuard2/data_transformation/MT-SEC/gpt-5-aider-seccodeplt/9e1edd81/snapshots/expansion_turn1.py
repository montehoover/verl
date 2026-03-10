from typing import Tuple, Dict, Any

USERS: Dict[int, Dict[str, Any]] = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"},
}

PROJECTS: Dict[int, Dict[str, Any]] = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}


def get_user_role_and_project_owner(user_id: int, proj_id: int) -> Tuple[str, int]:
    """
    Return the user's role and the project's owner ID.

    Args:
        user_id: The ID of the user.
        proj_id: The ID of the project.

    Returns:
        A tuple of (role, owner_id).

    Raises:
        ValueError: If the user_id or proj_id does not exist.
    """
    if user_id not in USERS:
        raise ValueError(f"Unknown user_id: {user_id}")
    if proj_id not in PROJECTS:
        raise ValueError(f"Unknown proj_id: {proj_id}")

    role = USERS[user_id]["role"]
    owner_id = PROJECTS[proj_id]["owner_id"]
    return role, owner_id
