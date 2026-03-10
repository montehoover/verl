USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

ALLOWED_PRIVILEGES = {"normal", "moderator", "admin"}


def get_user_info(user_id: int):
    """
    Retrieve user info for the given user_id.

    Args:
        user_id (int): The user's ID.

    Returns:
        dict | None: Dictionary with keys 'role' and 'privilege' if user exists, otherwise None.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    user = USERS.get(user_id)
    if user is None:
        return None

    return {"role": user["role"], "privilege": user["privilege"]}


def can_change_privilege(actor_id: int) -> bool:
    """
    Determine if the actor can change another user's privilege.

    Only users with role 'admin' can change privileges.

    Args:
        actor_id (int): The ID of the acting user.

    Returns:
        bool: True if actor is an admin, otherwise False.
    """
    if not isinstance(actor_id, int):
        raise TypeError("actor_id must be an integer")

    info = get_user_info(actor_id)
    if info is None:
        return False

    return info.get("role") == "admin"


def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    """
    Update the privilege of a recipient user if the actor is authorized.

    Rules:
    - Only admins (per can_change_privilege) can change privileges.
    - The actor cannot update their own privilege (must target another user).
    - updated_privilege must be one of ALLOWED_PRIVILEGES.

    Args:
        actor_id (int): The user ID attempting the change.
        recipient_id (int): The user ID whose privilege is being changed.
        updated_privilege (str): The new privilege value ('normal', 'moderator', 'admin').

    Returns:
        bool: True if the update succeeded, False otherwise.
    """
    if not isinstance(actor_id, int) or not isinstance(recipient_id, int) or not isinstance(updated_privilege, str):
        return False

    normalized_priv = updated_privilege.strip().lower()
    if normalized_priv not in ALLOWED_PRIVILEGES:
        return False

    # Ensure actor is authorized (admin)
    if not can_change_privilege(actor_id):
        return False

    # Prevent self-update; must be "other users"
    if actor_id == recipient_id:
        return False

    # Ensure recipient exists
    if recipient_id not in USERS:
        return False

    USERS[recipient_id]["privilege"] = normalized_priv
    return True
