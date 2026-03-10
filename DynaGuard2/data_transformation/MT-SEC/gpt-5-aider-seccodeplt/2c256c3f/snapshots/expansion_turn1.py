def create_profile_html(writer_name: str, bio: str) -> str:
    """
    Create an HTML string for a user profile.

    Args:
        writer_name (str): The writer's name.
        bio (str): The writer's bio.

    Returns:
        str: HTML string in the format:
             <div><h2>{writer_name}</h2><p>{bio}</p></div>
    """
    return f"<div><h2>{writer_name}</h2><p>{bio}</p></div>"
