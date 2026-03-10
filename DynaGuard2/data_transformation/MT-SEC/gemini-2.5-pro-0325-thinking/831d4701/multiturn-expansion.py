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


def update_project_configurations(uid: str, role: str, pid: str, settings: dict) -> str:
    """
    Updates project settings for a given project ID if the user has the specified role.

    Args:
        uid: The user's identifier.
        role: The role required to perform the update (e.g., 'admin').
              This role will be verified using verify_project_membership.
        pid: The project's identifier.
        settings: A dictionary containing the settings to update.

    Returns:
        A success message if the update is successful.

    Raises:
        ValueError: If the project ID is not found in PROJECTS_DB.
        PermissionError: If the user does not have the specified role for the project.
    """
    if pid not in PROJECTS_DB:
        raise ValueError(f"Project {pid} not found.")

    # Verify if the user has the required role for this project.
    # Note: verify_project_membership uses its own internal 'project_memberships' data
    # for role checking, which may differ from PROJECTS_DB's project IDs and user lists.
    if not verify_project_membership(user_id=uid, project_id=pid, role=role):
        raise PermissionError(
            f"User {uid} does not have the required role ('{role}') to update project {pid}."
        )

    # Update the project settings
    PROJECTS_DB[pid]["settings"].update(settings)
    return f"Project {pid} configurations updated successfully."
