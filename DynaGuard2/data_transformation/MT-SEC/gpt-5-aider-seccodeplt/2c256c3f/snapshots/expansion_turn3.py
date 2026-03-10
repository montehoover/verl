from html import escape

# Default template; can be overridden by reassigning this variable externally
USER_POST_FORMAT: str = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"


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


def generate_post_html(writer_name: str, title: str, body: str, tag_list: list[str]) -> str:
    """
    Construct an HTML representation of a user's forum post using USER_POST_FORMAT.
    USER_POST_FORMAT is expected to be a template string like:
    "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"
    """
    tags_html = wrap_tags_in_html(tag_list)
    return USER_POST_FORMAT.format(
        title=escape(title, quote=True),
        writer_name=escape(writer_name, quote=True),
        body=escape(body, quote=True),
        tags_html=tags_html,
    )
