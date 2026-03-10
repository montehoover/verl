def create_profile_html(writer_name: str, bio: str) -> str:
    """
    Generates an HTML string for a writer's profile.

    Args:
        writer_name: The name of the writer.
        bio: A short biography of the writer.

    Returns:
        An HTML string representing the writer's profile.
    """
    return f"<div><h2>{writer_name}</h2><p>{bio}</p></div>"
