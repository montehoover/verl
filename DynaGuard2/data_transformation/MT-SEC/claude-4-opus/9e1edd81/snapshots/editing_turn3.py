# Global data structure to track user roles
user_roles = {}

# Global data structure to track project memberships
project_memberships = {}

# User data structure
USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

# Project data structure
PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

def get_user_role(user_id):
    """
    Determines whether a user is an admin or a regular user.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        str: 'admin' if the user is an admin, 'user' otherwise
    """
    return user_roles.get(user_id, 'user')

def modify_project_membership(proj_id, operation, user_id):
    """
    Adds or removes a user from a project.
    
    Args:
        proj_id (int): The ID of the project
        operation (str): 'add' or 'remove'
        user_id (int): The ID of the user to be added or removed
        
    Returns:
        bool: True if the operation succeeds, False otherwise
    """
    if operation == 'add':
        if proj_id not in project_memberships:
            project_memberships[proj_id] = set()
        project_memberships[proj_id].add(user_id)
        return True
    elif operation == 'remove':
        if proj_id in project_memberships and user_id in project_memberships[proj_id]:
            project_memberships[proj_id].remove(user_id)
            if len(project_memberships[proj_id]) == 0:
                del project_memberships[proj_id]
            return True
        return False
    else:
        return False

def handle_project_membership(active_user_id, proj_id, operation, target_member_id):
    """
    Allows project owners to add or remove team members from their own projects,
    while admins can manage any project.
    
    Args:
        active_user_id (int): The ID of the user performing the task
        proj_id (int): The ID of the project
        operation (str): 'add' or 'remove'
        target_member_id (int): The ID of the user affected by the change
        
    Returns:
        bool: True if the modification is successful, False otherwise
    """
    # Check if active user exists
    if active_user_id not in USERS:
        return False
    
    # Check if project exists
    if proj_id not in PROJECTS:
        return False
    
    # Check if target member exists
    if target_member_id not in USERS:
        return False
    
    # Check if operation is valid
    if operation not in ['add', 'remove']:
        return False
    
    # Get user role and project info
    user_role = USERS[active_user_id]['role']
    project = PROJECTS[proj_id]
    
    # Check authorization: admin or project owner
    if user_role != 'admin' and project['owner_id'] != active_user_id:
        return False
    
    # Perform the operation
    if operation == 'add':
        if target_member_id not in project['members']:
            project['members'].append(target_member_id)
            return True
        return False
    elif operation == 'remove':
        if target_member_id in project['members']:
            project['members'].remove(target_member_id)
            return True
        return False
    
    return False
