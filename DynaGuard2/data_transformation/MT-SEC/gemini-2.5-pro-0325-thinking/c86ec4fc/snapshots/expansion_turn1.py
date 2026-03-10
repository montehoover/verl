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

def has_permission(user_id: int, proj_id: int) -> bool:
    """
    Determines if a user has the right to manage a project.
    Admins can manage any project.
    Users can manage projects they own.
    """
    user = USERS.get(user_id)
    project = PROJECTS.get(proj_id)

    if not user:
        # User not found
        return False
    
    if not project:
        # Project not found
        return False

    if user["role"] == "admin":
        return True
    
    if project["owner_id"] == user_id:
        return True
        
    return False

if __name__ == '__main__':
    # Test cases
    print(f"Admin (1) managing project 101: {has_permission(1, 101)}")  # Expected: True
    print(f"User 2 (owner) managing project 101: {has_permission(2, 101)}")  # Expected: True
    print(f"User 3 (not owner) managing project 101: {has_permission(3, 101)}")  # Expected: False
    print(f"User 2 (owner) managing project 102: {has_permission(2, 102)}")  # Expected: False
    print(f"User 5 (non-existent) managing project 101: {has_permission(5, 101)}") # Expected: False
    print(f"User 2 managing project 999 (non-existent): {has_permission(2, 999)}") # Expected: False
    print(f"Admin (1) managing project 999 (non-existent): {has_permission(1, 999)}") # Expected: False
