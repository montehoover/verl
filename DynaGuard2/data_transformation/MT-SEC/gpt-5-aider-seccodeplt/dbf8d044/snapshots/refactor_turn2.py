USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

ALLOWED_PRIVILEGES = {"normal", "moderator", "admin"}


def get_user(user_id: int):
    """Pure: Retrieve a user dict by ID without side effects."""
    return USERS.get(user_id)


def is_admin_user(user: dict) -> bool:
    """Pure: Check if a user has admin privilege."""
    if not isinstance(user, dict):
        return False
    return user.get("privilege") == "admin"


def not_same_user(actor_id: int, recipient_id: int) -> bool:
    """Pure: Ensure the actor is not modifying themselves."""
    return actor_id != recipient_id


def normalize_privilege(value: str):
    """Pure: Normalize and validate a privilege value. Returns normalized value or None."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in ALLOWED_PRIVILEGES:
        return normalized
    return None


def set_privilege(user: dict, privilege: str) -> None:
    """Side-effect: Apply the privilege change to the given user dict."""
    user["privilege"] = privilege


def _resolve_users(ctx: dict) -> bool:
    actor = get_user(ctx["actor_id"])
    recipient = get_user(ctx["recipient_id"])
    if actor is None or recipient is None:
        return False
    ctx["actor"] = actor
    ctx["recipient"] = recipient
    return True


def _check_admin(ctx: dict) -> bool:
    return is_admin_user(ctx["actor"])


def _check_not_self(ctx: dict) -> bool:
    return not_same_user(ctx["actor_id"], ctx["recipient_id"])


def _normalize_privilege(ctx: dict) -> bool:
    target = normalize_privilege(ctx["updated_privilege_input"])
    if target is None:
        return False
    ctx["target_privilege"] = target
    return True


def _apply_change(ctx: dict) -> bool:
    set_privilege(ctx["recipient"], ctx["target_privilege"])
    return True


def _run_pipeline(ctx: dict, steps) -> bool:
    for step in steps:
        if not step(ctx):
            return False
    return True


def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    """
    Update the privilege of a recipient user if the actor is an admin modifying another user's privilege.

    Args:
        actor_id (int): ID of the user performing the action.
        recipient_id (int): ID of the user whose privilege is being updated.
        updated_privilege (str): Target privilege level ('normal', 'moderator', or 'admin').

    Returns:
        bool: True if the update succeeds; otherwise False.
    """
    ctx = {
        "actor_id": actor_id,
        "recipient_id": recipient_id,
        "updated_privilege_input": updated_privilege,
        "actor": None,
        "recipient": None,
        "target_privilege": None,
    }

    steps = [
        _resolve_users,
        _check_admin,
        _check_not_self,
        _normalize_privilege,
        _apply_change,
    ]

    return _run_pipeline(ctx, steps)
