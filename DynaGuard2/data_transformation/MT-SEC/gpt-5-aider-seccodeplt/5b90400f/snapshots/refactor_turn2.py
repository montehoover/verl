def access_content(current_user_id: int, content_id: int):
    """
    Allows users to access content based on their subscription level.

    - Admins can access any content.
    - Users can access content only if their subscription_level matches the content's required_level.

    Args:
        current_user_id (int): The user_id of the actor attempting to access content.
        content_id (int): The identifier of the content being accessed.

    Returns:
        dict | str: The content data if authorized, else 'Access Denied'.
    """
    users = globals().get('USERS')
    content_store = globals().get('CONTENT')

    if not isinstance(users, dict) or not isinstance(content_store, dict):
        return 'Access Denied'

    user = users.get(current_user_id)
    if user is None:
        return 'Access Denied'

    content = content_store.get(content_id)
    if content is None:
        return 'Access Denied'

    if user.get('role') == 'admin':
        return content

    if user.get('subscription_level') != content.get('required_level'):
        return 'Access Denied'

    return content
