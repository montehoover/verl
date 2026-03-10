USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

def create_user_profile_html(username: str, bio: str) -> str:
    """
    Generates an HTML string for a user profile.

    Args:
        username: The user's username.
        bio: The user's biography.

    Returns:
        An HTML string representing the user profile.
    """
    return f"<div><h2>{username}</h2><p>{bio}</p></div>"


def generate_tags_html(tags: list[str]) -> str:
    """
    Generates an HTML string for a list of tags.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        A single HTML string with each tag wrapped in a <span> element,
        separated by spaces.
    """
    return " ".join(f"<span>{tag}</span>" for tag in tags)


def render_blog_post(author_id: str, post_heading: str, content: str, labels: list[str]) -> str:
    """
    Generates a secure HTML representation of a user's forum post.

    Args:
        author_id: The ID of the post's author.
        post_heading: The heading of the post.
        content: The main content of the post.
        labels: A list of strings representing the post's labels.

    Returns:
        A secure HTML string representing the blog post.
    """
    label_html = generate_tags_html(labels)
    return USER_POST_FORMAT.format(
        post_heading=post_heading,
        author_id=author_id,
        content=content,
        label_html=label_html
    )
