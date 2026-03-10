USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"

def _generate_tags_html(post_tags: list[str]) -> str:
    """
    Generates an HTML string for a list of tags.

    Args:
        post_tags: A list of tags.

    Returns:
        A string containing the HTML representation of the tags.
    """
    return "".join([f"<span>{tag}</span>" for tag in post_tags])

def build_post_html(user_name: str, post_heading: str, post_content: str, post_tags: list[str]) -> str:
    """
    Generates an HTML representation of a user's post for a public forum.

    Args:
        user_name: The author of the post.
        post_heading: The title of the user's post.
        post_content: The main content of the user's post.
        post_tags: A list of tags associated with the post.

    Returns:
        A string containing the sanitized HTML representation of the user's post.
    """
    tags_html = _generate_tags_html(post_tags)
    
    post_html = USER_POST_FORMAT.format(
        post_heading=post_heading,
        user_name=user_name,
        post_content=post_content,
        tags_html=tags_html
    )
    return post_html
