# Global data structure for user roles
# Example: {1: 'admin', 2: 'user', 3: 'admin'}
USER_ROLES = {}

def check_user_role(user_id: int) -> str:
    """
    Determines if a user is an admin or a regular user.

    Args:
        user_id: The ID of the user.

    Returns:
        'admin' if the user is an admin, or 'user' if they are a regular user.
        Returns 'user' by default if the user_id is not found or has no specific role.
    """
    role = USER_ROLES.get(user_id)
    if role == 'admin':
        return 'admin'
    return 'user'

def format_project_title(title: str) -> str:
    """
    Formats the project title to title case, with common stop words
    in lowercase unless they are the first word.

    Args:
        title: The project title string.

    Returns:
        The formatted project title string.
    """
    stop_words = {'and', 'of', 'in'}
    words = title.split()
    if not words:
        return ""

    formatted_words = [words[0].capitalize()]
    for word in words[1:]:
        if word.lower() in stop_words:
            formatted_words.append(word.lower())
        else:
            formatted_words.append(word.capitalize())
    
    return " ".join(formatted_words)
