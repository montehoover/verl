from html import escape

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"

def text_to_paragraph(text: str) -> str:
    """
    Convert a text string into an HTML paragraph by wrapping it in <p> tags.
    Special characters are escaped to ensure safe HTML output.
    """
    return f"<p>{escape(text, quote=True)}</p>"

def format_post_elements(title: str, bullets: list[str], paragraph: str) -> str:
    """
    Format a post's elements into HTML:
    - Title wrapped in <h2>
    - Bullet points wrapped in <ul>/<li>
    - Paragraph wrapped in <p>
    Special characters are escaped to ensure safe HTML output.
    """
    safe_title = escape(title, quote=True)
    list_items = "".join(f"<li>{escape(item, quote=True)}</li>" for item in bullets)
    ul_block = f"<ul>{list_items}</ul>"
    paragraph_block = text_to_paragraph(paragraph)
    return f"<h2>{safe_title}</h2>{ul_block}{paragraph_block}"

def build_post_html(user_name: str, post_heading: str, post_content: str, post_tags: list[str]) -> str:
    """
    Build a sanitized HTML representation of a user's post.

    Uses the global USER_POST_FORMAT template:
    "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"
    """
    safe_user_name = escape(user_name, quote=True)
    safe_heading = escape(post_heading, quote=True)
    safe_content = escape(post_content, quote=True)
    tags_html = "".join(f"<span>{escape(tag, quote=True)}</span>" for tag in post_tags)
    return USER_POST_FORMAT.format(
        user_name=safe_user_name,
        post_heading=safe_heading,
        post_content=safe_content,
        tags_html=tags_html,
    )
