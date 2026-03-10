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

def validate_project_access(user_id: str, proj_id: str, access_level: str) -> bool:
    """
    Validates if a user has the necessary access rights to a project.

    Args:
        user_id: The user's identifier.
        proj_id: The project's identifier.
        access_level: The required access level (e.g., "read", "write", "admin").

    Returns:
        True if the user has the required access, False otherwise.
    """
    # This is a placeholder for actual project access validation.
    # In a real application, you would check against a database
    # that stores user roles and permissions for each project.
    # For demonstration purposes, let's assume some hardcoded access rules.

    # Example: User "admin" has "admin" access to all projects.
    if user_id == "admin" and access_level == "admin":
        return True

    # Example: User "editor1" has "write" access to "project123".
    if user_id == "editor1" and proj_id == "project123" and access_level in ["read", "write"]:
        return True

    # Example: User "viewer1" has "read" access to "project456".
    if user_id == "viewer1" and proj_id == "project456" and access_level == "read":
        return True

    return False
