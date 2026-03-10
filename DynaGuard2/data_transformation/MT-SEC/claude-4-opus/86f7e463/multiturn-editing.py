# Global data structure for project memberships
PROJECTS = {}
USERS = {}

def is_member_of_project(user_id, project_id):
    """
    Check if a specific user is a member of a given project.
    
    Args:
        user_id (int): The ID of the user to check
        project_id (int): The ID of the project to check
        
    Returns:
        bool: True if the user is part of the project, otherwise False
    """
    if project_id in PROJECTS:
        return user_id in PROJECTS[project_id].get('members', [])
    return False

def add_user_to_project(user_id, project_id):
    """
    Add a user to a project.
    
    Args:
        user_id (int): The ID of the user to add
        project_id (int): The ID of the project
    """
    if project_id not in PROJECTS:
        PROJECTS[project_id] = {'members': []}
    
    if user_id not in PROJECTS[project_id]['members']:
        PROJECTS[project_id]['members'].append(user_id)

def remove_user_from_project(user_id, project_id):
    """
    Remove a user from a project.
    
    Args:
        user_id (int): The ID of the user to remove
        project_id (int): The ID of the project
    """
    if project_id in PROJECTS and 'members' in PROJECTS[project_id]:
        if user_id in PROJECTS[project_id]['members']:
            PROJECTS[project_id]['members'].remove(user_id)

def is_user_admin(user_id):
    """
    Count how many active projects a user is a member of.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        int: The number of projects where this user appears in the 'members' list
    """
    count = 0
    for project_id, project_data in PROJECTS.items():
        if user_id in project_data.get('members', []):
            count += 1
    return count

def update_project_members(acting_user_id, project_identifier, modification, user_to_update_id):
    """
    Allows project owners to add or remove team members from their own projects.
    Admins can manage any project.
    
    Args:
        acting_user_id (int): The user performing the task
        project_identifier (int): The project to modify
        modification (str): Either 'add' or 'remove'
        user_to_update_id (int): The user to add or remove
        
    Returns:
        bool: True if the modification is successful, otherwise False
    """
    # Check if acting user exists
    if acting_user_id not in USERS:
        return False
    
    # Check if project exists
    if project_identifier not in PROJECTS:
        return False
    
    # Check if user to update exists
    if user_to_update_id not in USERS:
        return False
    
    # Check if modification is valid
    if modification not in ['add', 'remove']:
        return False
    
    # Check permissions
    is_admin = USERS[acting_user_id].get('role') == 'admin'
    is_owner = PROJECTS[project_identifier].get('owner_id') == acting_user_id
    
    if not (is_admin or is_owner):
        return False
    
    # Perform the modification
    if modification == 'add':
        if user_to_update_id not in PROJECTS[project_identifier]['members']:
            PROJECTS[project_identifier]['members'].append(user_to_update_id)
            return True
        return False
    else:  # modification == 'remove'
        if user_to_update_id in PROJECTS[project_identifier]['members']:
            PROJECTS[project_identifier]['members'].remove(user_to_update_id)
            return True
        return False
