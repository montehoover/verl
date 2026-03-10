def access_content(current_user_id: int, content_id: int) -> str:
    """
    Determine if the user can access the content.
    - Admins can access any content.
    - Users can access content only if their subscription_level matches the content's required_level.

    Returns:
        The content title (string) if authorized, else 'Access Denied'.

    Notes:
        Expects global dictionaries USERS and CONTENT to be defined, e.g.:
        USERS = {
            1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
            ...
        }
        CONTENT = {
            100: {"title": "Basic Content 1", "required_level": "basic"},
            ...
        }
    """
    # Access global data stores for users and content
    try:
        users = USERS  # type: ignore[name-defined]
        content_store = CONTENT  # type: ignore[name-defined]
    except NameError:
        # Required globals are not available
        return "Access Denied"

    user = users.get(current_user_id)
    content = content_store.get(content_id)

    # Validate presence of user and content
    if not user or not content:
        return "Access Denied"

    role = user.get("role")
    subscription_level = user.get("subscription_level")
    required_level = content.get("required_level")

    # Admins can access any content
    if role == "admin":
        return content.get("title", "Access Denied")

    # Regular users must match subscription level to required level
    if role == "user" and subscription_level == required_level:
        return content.get("title", "Access Denied")

    # All other cases are denied
    return "Access Denied"
