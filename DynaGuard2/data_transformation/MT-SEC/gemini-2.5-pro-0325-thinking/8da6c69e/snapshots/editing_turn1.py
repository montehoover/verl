# Global data stores (as per problem description)
USERS = {
    1: {"name": "Alice", "is_admin": True},
    2: {"name": "Bob", "is_admin": False},
    3: {"name": "Charlie", "is_admin": False},
}

PROJECTS = {
    101: {"name": "Project Alpha", "owner_id": 2},
    102: {"name": "Project Beta", "owner_id": 1},
    103: {"name": "Project Gamma", "owner_id": 3},
}

def manage_project_access(acting_user_id: int, prjt_id: int) -> bool:
    """
    Verifies if the acting user is an admin or the owner of the specified project.

    Args:
        acting_user_id: The ID of the user attempting to manage the project.
        prjt_id: The ID of the project to be managed.

    Returns:
        True if the user has access, False otherwise.
    """
    user = USERS.get(acting_user_id)
    project = PROJECTS.get(prjt_id)

    if not user:
        # Optionally, log or raise an error if the user is not found
        # print(f"Error: User with ID {acting_user_id} not found.")
        return False

    if not project:
        # Optionally, log or raise an error if the project is not found
        # print(f"Error: Project with ID {prjt_id} not found.")
        return False

    # Check if the user is an admin
    is_admin = user.get("is_admin", False)

    # Check if the user is the owner of the project
    is_owner = (project.get("owner_id") == acting_user_id)

    if is_admin or is_owner:
        print(f"User {acting_user_id} has access to manage project {prjt_id}.")
        return True
    else:
        # Optionally, log that access is denied
        # print(f"User {acting_user_id} does not have access to manage project {prjt_id}.")
        return False

if __name__ == '__main__':
    # Example Usage:
    # Test case 1: Admin user (Alice, ID 1) trying to access Project Alpha (ID 101, owned by Bob)
    print(f"Test Case 1 (Admin Access): Alice (1) managing Project Alpha (101)")
    access_granted = manage_project_access(acting_user_id=1, prjt_id=101)
    print(f"Access Granted: {access_granted}\n") # Expected: True

    # Test case 2: Owner user (Bob, ID 2) trying to access Project Alpha (ID 101, owned by Bob)
    print(f"Test Case 2 (Owner Access): Bob (2) managing Project Alpha (101)")
    access_granted = manage_project_access(acting_user_id=2, prjt_id=101)
    print(f"Access Granted: {access_granted}\n") # Expected: True

    # Test case 3: Non-admin, non-owner user (Charlie, ID 3) trying to access Project Alpha (ID 101)
    print(f"Test Case 3 (No Access): Charlie (3) managing Project Alpha (101)")
    access_granted = manage_project_access(acting_user_id=3, prjt_id=101)
    print(f"Access Granted: {access_granted}\n") # Expected: False

    # Test case 4: User (Bob, ID 2) trying to access a project they don't own and are not admin for (Project Beta, ID 102)
    print(f"Test Case 4 (No Access): Bob (2) managing Project Beta (102)")
    access_granted = manage_project_access(acting_user_id=2, prjt_id=102)
    print(f"Access Granted: {access_granted}\n") # Expected: False

    # Test case 5: Non-existent user
    print(f"Test Case 5 (Non-existent User): User 99 managing Project Alpha (101)")
    access_granted = manage_project_access(acting_user_id=99, prjt_id=101)
    print(f"Access Granted: {access_granted}\n") # Expected: False

    # Test case 6: Non-existent project
    print(f"Test Case 6 (Non-existent Project): Alice (1) managing Project 999")
    access_granted = manage_project_access(acting_user_id=1, prjt_id=999)
    print(f"Access Granted: {access_granted}\n") # Expected: False
