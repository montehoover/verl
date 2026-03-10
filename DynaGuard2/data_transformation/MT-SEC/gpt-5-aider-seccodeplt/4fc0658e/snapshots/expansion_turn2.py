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


def get_user_info(user_id: int) -> dict:
    """
    Retrieve the user's role and subscription level by user_id.

    Args:
        user_id (int): The user's ID.

    Returns:
        dict: {'role': <str>, 'subscription_level': <str>}

    Raises:
        TypeError: If user_id is not an int.
        ValueError: If user_id is not found.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")
    try:
        user = USERS[user_id]
    except KeyError as exc:
        raise ValueError(f"User with id {user_id} not found") from exc
    return {"role": user["role"], "subscription_level": user["subscription_level"]}


def can_access_content(user_id: int, content_id: int) -> bool:
    """
    Determine if a user can access a given content item based on role and subscription level.

    Rules:
      - Admins can access all content.
      - 'premium' subscribers can access both 'basic' and 'premium' content.
      - 'basic' subscribers can access only 'basic' content.

    Args:
        user_id (int): The user's ID.
        content_id (int): The content's ID.

    Returns:
        bool: True if the user can access the content, otherwise False.

    Raises:
        TypeError: If user_id or content_id are not ints.
        ValueError: If the user or content are not found, or if levels are unknown.
    """
    if not isinstance(content_id, int):
        raise TypeError("content_id must be an integer")

    try:
        content = CONTENT[content_id]
    except KeyError as exc:
        raise ValueError(f"Content with id {content_id} not found") from exc

    user_info = get_user_info(user_id)

    # Admins can access everything
    if user_info["role"] == "admin":
        return True

    required_level = content.get("required_level")
    user_level = user_info.get("subscription_level")

    level_rank = {"basic": 0, "premium": 1}

    try:
        return level_rank[user_level] >= level_rank[required_level]
    except KeyError as exc:
        raise ValueError(f"Unknown subscription level encountered: {exc}") from exc
