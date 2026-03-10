def generate_announcement(title: str, message: str) -> str:
    """
    Creates a basic shell script command for announcing updates or news.

    Args:
        title: The headline of the announcement.
        message: The body of the announcement.

    Returns:
        A simple echo string for the shell.
    """
    # Escape single quotes in title and message to prevent shell injection issues
    # and ensure the strings are correctly interpreted by echo.
    safe_title = title.replace("'", "'\\''")
    safe_message = message.replace("'", "'\\''")
    return f"echo '{safe_title}\n\n{safe_message}'"
