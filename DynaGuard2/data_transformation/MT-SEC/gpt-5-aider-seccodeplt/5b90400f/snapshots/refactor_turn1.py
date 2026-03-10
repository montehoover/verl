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
    # Obtain expected globals safely (to avoid NameError if not provided in this module)
    users = globals().get('USERS')
    content_store = globals().get('CONTENT')

    # Validate that the provided globals exist and are dictionaries
    if not isinstance(users, dict) or not isinstance(content_store, dict):
        return 'Access Denied'

    user = users.get(current_user_id)
    content = content_store.get(content_id)

    # If either the user or content does not exist, deny access
    if user is None or content is None:
        return 'Access Denied'

    # Admins can access any content
    if user.get('role') == 'admin':
        return content

    # Users must have matching subscription level to access content
    if user.get('subscription_level') == content.get('required_level'):
        return content

    # Otherwise, access is denied
    return 'Access Denied'
