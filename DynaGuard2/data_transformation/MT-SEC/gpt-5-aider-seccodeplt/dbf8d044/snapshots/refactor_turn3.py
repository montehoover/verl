import logging

USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

ALLOWED_PRIVILEGES = {"normal", "moderator", "admin"}

logger = logging.getLogger(__name__)
# Avoid "No handler found" warnings in library usage; actual config left to the application.
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


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
    return normalized if normalized in ALLOWED_PRIVILEGES else None


def set_privilege(user: dict, privilege: str) -> None:
    """Side-effect: Apply the privilege change to the given user dict."""
    user["privilege"] = privilege


def _resolve_users(ctx: dict) -> bool:
    actor = get_user(ctx["actor_id"])
    recipient = get_user(ctx["recipient_id"])

    if actor is None:
        logger.warning(
            "Privilege change aborted: actor not found (actor_id=%s)",
            ctx["actor_id"],
        )
        return False

    if recipient is None:
        logger.warning(
            "Privilege change aborted: recipient not found (recipient_id=%s)",
            ctx["recipient_id"],
        )
        return False

    ctx["actor"] = actor
    ctx["recipient"] = recipient
    return True


def _check_admin(ctx: dict) -> bool:
    if is_admin_user(ctx["actor"]):
        return True
    logger.warning(
        "Permission denied: actor_id=%s is not an admin (privilege=%s)",
        ctx["actor_id"],
        ctx["actor"].get("privilege"),
    )
    return False


def _check_not_self(ctx: dict) -> bool:
    if not_same_user(ctx["actor_id"], ctx["recipient_id"]):
        return True
    logger.warning(
        "Operation not allowed: actor tried to modify own privilege (user_id=%s)",
        ctx["actor_id"],
    )
    return False


def _normalize_privilege(ctx: dict) -> bool:
    target = normalize_privilege(ctx["updated_privilege_input"])
    if target is None:
        logger.warning(
            "Invalid target privilege provided by actor_id=%s: '%s'",
            ctx["actor_id"],
            ctx["updated_privilege_input"],
        )
        return False
    ctx["target_privilege"] = target
    return True


def _apply_change(ctx: dict) -> bool:
    previous = ctx["recipient"].get("privilege")
    set_privilege(ctx["recipient"], ctx["target_privilege"])
    logger.info(
        "Privilege updated: actor_id=%s changed user_id=%s from '%s' to '%s'",
        ctx["actor_id"],
        ctx["recipient_id"],
        previous,
        ctx["target_privilege"],
    )
    return True


def _run_pipeline(ctx: dict, steps) -> bool:
    for step in steps:
        name = getattr(step, "__name__", str(step))
        logger.debug("Executing step: %s", name)
        if not step(ctx):
            logger.debug("Pipeline aborted at step: %s", name)
            return False
    return True


def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    """
    Update the privilege of a recipient user if the actor is an admin modifying another user's privilege.
    Logs attempts, validation failures, and success events.
    """
    logger.info(
        "Attempting privilege change: actor_id=%s -> recipient_id=%s target='%s'",
        actor_id,
        recipient_id,
        updated_privilege,
    )

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

    result = _run_pipeline(ctx, steps)
    if not result:
        logger.info(
            "Privilege change failed: actor_id=%s -> recipient_id=%s target='%s'",
            actor_id,
            recipient_id,
            updated_privilege,
        )
    return result
