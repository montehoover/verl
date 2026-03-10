def create_user_profile_html(username: str, bio: str) -> str:
    """
    Generates an HTML string for a user profile.

    Args:
        username: The user's username.
        bio: The user's biography.

    Returns:
        An HTML string representing the user's profile.
    """
    return f"<div><h2>{username}</h2><p>{bio}</p></div>"


def generate_tags_html(tags: list[str]) -> str:
    """
    Generates an HTML string for a list of tags.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        A single string where each tag is wrapped in a <span> element,
        separated by spaces.
    """
    return " ".join(f"<span>{tag}</span>" for tag in tags)


USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"


def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    """
    Generates a full HTML view of a user's post.

    Args:
        username: The user's username.
        post_title: The title of the post.
        post_body: The main content of the post.
        tags: A list of tags associated with the post.

    Returns:
        A complete HTML string representing the user's post.
    """
    tags_html = generate_tags_html(tags)
    return USER_POST_FORMAT.format(
        post_title=post_title,
        username=username,
        post_body=post_body,
        tags_html=tags_html
    )
