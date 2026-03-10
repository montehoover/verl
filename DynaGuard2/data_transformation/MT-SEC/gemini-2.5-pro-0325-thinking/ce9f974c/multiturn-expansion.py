PROJECTS_DB = {
    "PROJ001": {
        "project_id": "PROJ001",
        "creator_id": "USER1",
        "members": ["USER1", "USER2", "USER3"],
        "settings": {"visibility": "private", "deadline": "2023-12-31"}
    },
    "PROJ002": {
        "project_id": "PROJ002",
        "creator_id": "USER4",
        "members": ["USER4", "USER5"],
        "settings": {"visibility": "public", "deadline": "2023-10-15"}
    }
}

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

def change_project_settings(owner_id: str, access_level: str, proj_id: str, modifications: dict) -> str:
    """
    Allows authorized users to update project settings.

    Args:
        owner_id: The ID of the user attempting the change.
        access_level: The access level required for this operation (e.g., "admin", "write").
        proj_id: The ID of the project to modify.
        modifications: A dictionary containing the settings to update.

    Returns:
        A success message if the operation is successful.

    Raises:
        PermissionError: If the user does not have the required access level.
        ValueError: If the project ID is not found in PROJECTS_DB.
    """
    if not validate_project_access(user_id=owner_id, proj_id=proj_id, access_level=access_level):
        raise PermissionError(f"User {owner_id} does not have '{access_level}' access to project {proj_id}.")

    if proj_id not in PROJECTS_DB:
        raise ValueError(f"Project {proj_id} not found.")

    PROJECTS_DB[proj_id]["settings"].update(modifications)
    return f"Project {proj_id} settings updated successfully by user {owner_id}."
