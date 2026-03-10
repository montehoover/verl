USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

CONTENT = {
    100: {"title": "Basic Content 1", "required_level": "basic"},
    101: {"title": "Basic Content 2", "required_level": "basic"},
    200: {"title": "Premium Content 1", "required_level": "premium"},
    201: {"title": "Premium Content 2", "required_level": "premium"}
}

LEVEL_ORDER = {"basic": 0, "premium": 1}


def is_admin(user: dict) -> bool:
    """Return True if the user has an admin role."""
    return (user.get("role") or "").strip().lower() == "admin"


def has_subscription_access(subscription_level: str, required_level: str) -> bool:
    """
    Return True if a subscription level satisfies a required level.
    Assumes LEVEL_ORDER defines the hierarchy.
    """
    sub = (subscription_level or "").strip().lower()
    req = (required_level or "").strip().lower()

    if sub not in LEVEL_ORDER or req not in LEVEL_ORDER:
        return False

    return LEVEL_ORDER[sub] >= LEVEL_ORDER[req]


def can_user_access_content(user: dict, content: dict) -> bool:
    """
    Pure function that determines if a given user can access a given content.
    Admins have unrestricted access; regular users must meet required level.
    """
    if not user or not content:
        return False

    if is_admin(user):
        return True

    return has_subscription_access(
        user.get("subscription_level"),
        content.get("required_level"),
    )


def fetch_content(given_user_id: int, target_content_id: int) -> str:
    """
    Manages content access based on the user's role and subscription status.

    Returns the content title if permitted; otherwise, 'Access Denied'.
    """
    user = USERS.get(given_user_id)
    content = CONTENT.get(target_content_id)

    if not can_user_access_content(user, content):
        return "Access Denied"

    return content.get("title", "Access Denied")
