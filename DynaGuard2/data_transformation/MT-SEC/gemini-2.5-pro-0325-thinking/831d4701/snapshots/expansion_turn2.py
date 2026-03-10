def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if the credentials are correct, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, you would check credentials against a database
    # or another secure authentication mechanism.
    # For demonstration purposes, let's assume a hardcoded user.
    if user_id == "admin" and password == "password123":
        return True
    return False


def verify_project_membership(user_id: str, project_id: str, role: str) -> bool:
    """
    Verifies if a user has a specific role within a project.

    Args:
        user_id: The user's identifier.
        project_id: The project's identifier.
        role: The role to verify (e.g., 'admin', 'editor', 'viewer').

    Returns:
        True if the user has the specified role in the project, False otherwise.
    """
    # This is a placeholder for actual project membership and role verification.
    # In a real application, you would query a database or another data source
    # to check the user's role in the specified project.
    # For demonstration purposes, let's assume some hardcoded project memberships.
    project_memberships = {
        "project123": {
            "userA": "admin",
            "userB": "editor",
        },
        "project456": {
            "userA": "viewer",
            "userC": "admin",
        }
    }

    if project_id in project_memberships:
        project_users = project_memberships[project_id]
        if user_id in project_users and project_users[user_id] == role:
            return True
    return False
