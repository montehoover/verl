import html

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"

def create_user_profile_html(user_name: str, user_bio: str) -> str:
    """
    Generates an HTML string for a user profile.

    Args:
        user_name: The name of the user.
        user_bio: The biography of the user.

    Returns:
        An HTML string representing the user profile.
    """
    return f"<div><h2>{user_name}</h2><p>{user_bio}</p></div>"


def generate_tags_html(tags: list[str]) -> str:
    """
    Generates an HTML string for a list of tags.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        A single string where each tag is enclosed in a <span> element,
        separated by spaces.
    """
    return " ".join(f"<span>{html.escape(tag)}</span>" for tag in tags)


def build_post_html(user_name: str, post_heading: str, post_content: str, post_tags: list[str]) -> str:
    """
    Generates an HTML representation of a user's post.

    Args:
        user_name: The name of the user.
        post_heading: The heading of the post.
        post_content: The content of the post.
        post_tags: A list of tags associated with the post.

    Returns:
        A sanitized HTML string representing the user's post.
    """
    tags_html = generate_tags_html(post_tags)
    
    # Sanitize inputs before formatting
    safe_user_name = html.escape(user_name)
    safe_post_heading = html.escape(post_heading)
    safe_post_content = html.escape(post_content)
    # tags_html is already sanitized by generate_tags_html

    return USER_POST_FORMAT.format(
        post_heading=safe_post_heading,
        user_name=safe_user_name,
        post_content=safe_post_content,
        tags_html=tags_html
    )
