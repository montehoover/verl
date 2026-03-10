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


def wrap_tags_in_html(tags: list[str]) -> str:
    """
    Wraps a list of tags in <span> HTML elements, separated by spaces.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        A single string with each tag wrapped in a <span> element,
        joined by spaces.
    """
    span_tags = [f"<span>{tag}</span>" for tag in tags]
    return " ".join(span_tags)
