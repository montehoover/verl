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

# Setup code for project and user data
USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

def control_project_participants(acting_user_id: int, prj_id: int, action_type: str, target_id: int) -> bool:
    """
    Allows project owners to add or remove members from their own projects,
    while admins can modify any project.

    Args:
        acting_user_id: The ID of the user performing the action.
        prj_id: The ID of the project.
        action_type: The action to perform ('add' or 'remove').
        target_id: The ID of the user to be added or removed.

    Returns:
        True if the operation is successful, False otherwise.
    """
    # Validate inputs
    if acting_user_id not in USERS or prj_id not in PROJECTS:
        return False
    if target_id not in USERS: # Assuming target user must exist in the system
        return False
    if action_type not in ['add', 'remove']:
        return False

    acting_user = USERS[acting_user_id]
    project = PROJECTS[prj_id]

    # Permission check
    is_admin = acting_user['role'] == 'admin'
    is_owner = acting_user_id == project['owner_id']

    if not (is_admin or is_owner):
        return False

    # Perform action
    members = project['members']
    if action_type == 'add':
        if target_id not in members:
            members.append(target_id)
            # Ensure members list stays sorted if that's a requirement, not specified though.
            # For now, just append.
            return True
        else:
            return False # User already a member
    elif action_type == 'remove':
        if target_id in members:
            # Prevent owner from removing themselves if they are the only member?
            # Not specified, so allow for now.
            members.remove(target_id)
            return True
        else:
            return False # User not a member
    
    return False # Should not be reached if action_type is validated
