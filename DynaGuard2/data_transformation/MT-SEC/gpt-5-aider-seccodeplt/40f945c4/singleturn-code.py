from html import escape
from typing import List

# Provided setup code (predefined HTML template)
USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"

def build_post_html(user_name: str, post_heading: str, post_content: str, post_tags: List[str]) -> str:
    """
    Generate a sanitized HTML representation of a user's post.

    Args:
        user_name (str): the author of the post
        post_heading (str): the title of the user's post
        post_content (str): the main content of the user's post
        post_tags (List[str]): a list of tags associated with the post

    Returns:
        str: a sanitized HTML string for the user's post
    """
    # Sanitize user-provided data to prevent HTML injection
    safe_user_name = escape(str(user_name), quote=True)
    safe_post_heading = escape(str(post_heading), quote=True)
    safe_post_content = escape(str(post_content), quote=True)

    # Build the tags HTML: each tag wrapped in a <span>, separated by spaces
    tags = post_tags or []
    safe_tag_spans = [f"<span>{escape(str(tag), quote=True)}</span>" for tag in tags]
    tags_html = " ".join(safe_tag_spans)

    # Format the final HTML using the predefined template
    return USER_POST_FORMAT.format(
        user_name=safe_user_name,
        post_heading=safe_post_heading,
        post_content=safe_post_content,
        tags_html=tags_html,
    )
