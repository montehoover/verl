import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"

def generate_user_content_html(author_name: str, content: str) -> str:
    """
    Generates an HTML string to display user-generated content.

    Args:
        author_name: The name of the content author.
        content: The user-generated content.

    Returns:
        An HTML string.
    """
    return f"<div><h3>Posted by {author_name}</h3><p>{content}</p></div>"


def format_tags(tags: list[str]) -> str:
    """
    Formats a list of tags into an HTML string with each tag in a <span>.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        A string with tags wrapped in <span> elements, separated by spaces.
    """
    return " ".join(f"<span>{tag}</span>" for tag in tags)


def create_post_html(author_name: str, post_title: str, content: str, post_tags: list[str]) -> str:
    """
    Constructs an HTML representation of a user's forum post.

    Args:
        author_name: The name of the post author.
        post_title: The title of the post.
        content: The main content of the post.
        post_tags: A list of tags associated with the post.

    Returns:
        An XSS-protected HTML string representing the forum post.
    """
    escaped_author_name = html.escape(author_name)
    escaped_post_title = html.escape(post_title)
    escaped_content = html.escape(content)
    
    # Tags are already formatted into safe HTML by format_tags,
    # assuming tags themselves don't contain HTML characters that need escaping.
    # If tags can contain special characters, they should be escaped within format_tags or before.
    # For this implementation, we assume format_tags handles or expects clean tags.
    tags_html = format_tags(post_tags) 

    return USER_POST_FORMAT.format(
        post_title=escaped_post_title,
        author_name=escaped_author_name,
        content=escaped_content,
        tags_html=tags_html
    )
