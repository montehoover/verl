# Global data structure to track user roles
user_roles = {}

# User and project data structures
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

def check_user_role(user_id):
    """Determines if a user is an admin or a regular user.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        str: 'admin' if the user is an admin, 'user' if they are a regular user
    """
    return user_roles.get(user_id, 'user')

def format_project_title(title):
    """Formats a project title in title case with common stop words lowercase.
    
    Args:
        title (str): The project title to format
        
    Returns:
        str: The formatted title with proper capitalization
    """
    stop_words = {'and', 'of', 'in', 'the', 'a', 'an', 'to', 'for', 'with', 'on', 'at', 'by', 'or'}
    words = title.split()
    
    formatted_words = []
    for i, word in enumerate(words):
        # Always capitalize the first word, otherwise check if it's a stop word
        if i == 0 or word.lower() not in stop_words:
            formatted_words.append(word.capitalize())
        else:
            formatted_words.append(word.lower())
    
    return ' '.join(formatted_words)

def control_project_participants(acting_user_id, prj_id, action_type, target_id):
    """Allows project owners to add/remove members from their projects, admins can modify any project.
    
    Args:
        acting_user_id (int): The ID of the user performing the action
        prj_id (int): The ID of the project
        action_type (str): Either 'add' or 'remove'
        target_id (int): The ID of the user to add or remove
        
    Returns:
        bool: True if the operation is successful, False otherwise
    """
    # Check if the project exists
    if prj_id not in PROJECTS:
        return False
    
    # Check if the acting user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if the target user exists
    if target_id not in USERS:
        return False
    
    # Check if action_type is valid
    if action_type not in ['add', 'remove']:
        return False
    
    # Get the acting user's role
    acting_user_role = USERS[acting_user_id]["role"]
    
    # Get the project
    project = PROJECTS[prj_id]
    
    # Check if the acting user has permission
    # Admins can modify any project, owners can modify their own projects
    if acting_user_role != "admin" and project["owner_id"] != acting_user_id:
        return False
    
    # Perform the action
    if action_type == 'add':
        if target_id not in project["members"]:
            project["members"].append(target_id)
            return True
        else:
            return False  # User already in project
    else:  # action_type == 'remove'
        if target_id in project["members"]:
            project["members"].remove(target_id)
            return True
        else:
            return False  # User not in project
