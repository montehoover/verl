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

def manage_team_access(executor_user_id, proj_id, operation, target_member_id):
    # Check if project exists
    if proj_id not in PROJECTS:
        return False
    
    # Check if executor user exists
    if executor_user_id not in USERS:
        return False
    
    # Check if target member exists
    if target_member_id not in USERS:
        return False
    
    # Check if operation is valid
    if operation not in ['add', 'remove']:
        return False
    
    # Get executor user info
    executor = USERS[executor_user_id]
    project = PROJECTS[proj_id]
    
    # Check permissions: admin can modify any project, owner can modify their own project
    if executor['role'] != 'admin' and project['owner_id'] != executor_user_id:
        return False
    
    # Perform the operation
    if operation == 'add':
        if target_member_id not in project['members']:
            project['members'].append(target_member_id)
            return True
        else:
            return False  # Member already exists
    else:  # operation == 'remove'
        if target_member_id in project['members']:
            project['members'].remove(target_member_id)
            return True
        else:
            return False  # Member doesn't exist
