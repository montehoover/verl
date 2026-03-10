# Global data structure to track user roles
user_roles = {}

# Global data structure to track project memberships
project_memberships = {}

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
