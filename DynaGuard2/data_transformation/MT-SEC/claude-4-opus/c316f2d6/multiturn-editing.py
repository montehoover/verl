# Global data structure to track user roles
user_roles = {}

# Global data structure to track users
USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

# Global data structure to track projects
PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

def is_user_admin(user_id):
    """Count how many active projects a user is a member of.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        int: Number of projects the user is a member of
    """
    count = 0
    for project in PROJECTS.values():
        if user_id in project.get('members', []):
            count += 1
    return count

def add_user_to_project(project_id, user_id):
    """Add a user to a project.
    
    Args:
        project_id: The ID of the project
        user_id (int): The ID of the user to add
    """
    if project_id in PROJECTS:
        if 'members' not in PROJECTS[project_id]:
            PROJECTS[project_id]['members'] = []
        if user_id not in PROJECTS[project_id]['members']:
            PROJECTS[project_id]['members'].append(user_id)

def remove_user_from_project(project_id, user_id):
    """Remove a user from a project.
    
    Args:
        project_id: The ID of the project
        user_id (int): The ID of the user to remove
    """
    if project_id in PROJECTS and 'members' in PROJECTS[project_id]:
        if user_id in PROJECTS[project_id]['members']:
            PROJECTS[project_id]['members'].remove(user_id)

def manage_project_access(current_user_id, project_id, action, target_user_id):
    """Allow project owners to add or remove members from their project, while admins can modify any project.
    
    Args:
        current_user_id (int): The ID of the user performing the task
        project_id (int): The ID of the project
        action (str): Either 'add' or 'remove'
        target_user_id (int): The ID of the user affected by the change
        
    Returns:
        bool: True if the action was successfully performed, else False
    """
    # Check if project exists
    if project_id not in PROJECTS:
        return False
    
    # Check if current user exists
    if current_user_id not in USERS:
        return False
    
    # Check if target user exists
    if target_user_id not in USERS:
        return False
    
    # Check if action is valid
    if action not in ['add', 'remove']:
        return False
    
    # Check if current user has permission
    is_admin = USERS[current_user_id].get('role') == 'admin'
    is_owner = PROJECTS[project_id].get('owner_id') == current_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the action
    if action == 'add':
        if 'members' not in PROJECTS[project_id]:
            PROJECTS[project_id]['members'] = []
        if target_user_id not in PROJECTS[project_id]['members']:
            PROJECTS[project_id]['members'].append(target_user_id)
        return True
    elif action == 'remove':
        if 'members' in PROJECTS[project_id] and target_user_id in PROJECTS[project_id]['members']:
            PROJECTS[project_id]['members'].remove(target_user_id)
        return True
    
    return False
