def verify_user_token(user_token: str) -> bool:
    """
    Verifies a user token.

    Args:
        user_token: The token string to verify.

    Returns:
        True if the token is valid, False otherwise.
    """
    # For now, let's assume any non-empty token is valid.
    # In a real application, this would involve checking against a database
    # or a token validation service.
    return bool(user_token)


def check_user_role(user_id: str, project_id: str) -> str:
    """
    Checks the user's role in a specific project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.

    Returns:
        The user's role in the project (e.g., "admin", "editor", "viewer").
    """
    # This is a placeholder. In a real application, this function would
    # query a database or another service to determine the user's role
    # for the given project.
    # Example:
    # if user_id == "user123" and project_id == "project_abc":
    #     return "admin"
    # elif user_id == "user456" and project_id == "project_abc":
    #     return "editor"
    # else:
    #     return "viewer"
    return "viewer"  # Default role
