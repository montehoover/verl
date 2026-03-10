import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

def create_user_html(author_name: str, user_bio: str) -> str:
    """
    Generates an HTML string displaying user information.

    Args:
        author_name: The name of the user.
        user_bio: A short biography of the user.

    Returns:
        An HTML string with the user's information.
    """
    return f"<div><h2>{author_name}</h2><p>{user_bio}</p></div>"


def generate_tags_html(tags: list[str]) -> str:
    """
    Generates an HTML string for a list of tags.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        A single HTML string with each tag wrapped in a <span> element,
        separated by spaces.
    """
    span_tags = [f"<span>{html.escape(tag)}</span>" for tag in tags]
    return " ".join(span_tags)


def render_forum_post(author_name: str, title: str, body_content: str, tag_list: list[str]) -> str:
    """
    Generates a secure HTML representation of a user's forum post.

    Args:
        author_name: The name of the post's author.
        title: The title of the post.
        body_content: The main content of the post.
        tag_list: A list of tags associated with the post.

    Returns:
        An XSS-protected HTML string representing the forum post.
    """
    safe_author_name = html.escape(author_name)
    safe_title = html.escape(title)
    safe_body_content = html.escape(body_content)
    
    tags_html = generate_tags_html(tag_list) # Tags are escaped in generate_tags_html

    return USER_POST_FORMAT.format(
        title=safe_title,
        author_name=safe_author_name,
        body_content=safe_body_content,
        tags_html=tags_html
    )
