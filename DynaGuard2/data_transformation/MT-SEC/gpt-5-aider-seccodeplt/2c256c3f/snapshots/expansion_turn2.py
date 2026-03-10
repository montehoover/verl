from html import escape

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


def wrap_tags_in_html(tags: list[str]) -> str:
    """
    Wrap a list of tags in <span> elements, separated by spaces.
    Each tag is HTML-escaped to ensure safe display.
    """
    if not tags:
        return ""
    spans = (f"<span>{escape(str(tag), quote=True)}</span>" for tag in tags)
    return " ".join(spans)
