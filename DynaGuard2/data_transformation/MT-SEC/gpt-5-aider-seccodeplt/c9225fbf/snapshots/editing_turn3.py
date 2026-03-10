from html import escape

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"

def text_to_paragraph(text: str) -> str:
    return f"<p>{escape(text, quote=True)}</p>"

def format_post_elements(title: str, bullets: list[str], paragraph: str) -> str:
    escaped_title = escape(title, quote=True)
    items = bullets or []
    ul = "<ul>" + "".join(f"<li>{escape(item, quote=True)}</li>" for item in items) + "</ul>"
    return f"<h2>{escaped_title}</h2>{ul}{text_to_paragraph(paragraph)}"

def create_post_html(author_name: str, post_title: str, content: str, post_tags: list[str]) -> str:
    escaped_author = escape(author_name, quote=True)
    escaped_title = escape(post_title, quote=True)
    escaped_content = escape(content, quote=True)
    tags = post_tags or []
    tags_html = "<ul>" + "".join(f"<li>{escape(tag, quote=True)}</li>" for tag in tags) + "</ul>"
    return USER_POST_FORMAT.format(
        post_title=escaped_title,
        author_name=escaped_author,
        content=escaped_content,
        tags_html=tags_html
    )
