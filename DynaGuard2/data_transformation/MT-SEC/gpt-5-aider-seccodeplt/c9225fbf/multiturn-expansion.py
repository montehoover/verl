import html

# Fallback template if USER_POST_FORMAT is not defined elsewhere in the project
if 'USER_POST_FORMAT' not in globals():
    USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"

def generate_user_content_html(author_name: str, content: str) -> str:
    return f"<div><h3>Posted by {author_name}</h3><p>{content}</p></div>"

def format_tags(tags: list[str]) -> str:
    return " ".join(f"<span>{tag}</span>" for tag in tags)

def create_post_html(author_name: str, post_title: str, content: str, post_tags: list[str]) -> str:
    safe_author = html.escape(author_name, quote=True)
    safe_title = html.escape(post_title, quote=True)
    safe_content = html.escape(content, quote=True)
    safe_tags_html = format_tags([html.escape(tag, quote=True) for tag in post_tags])
    return USER_POST_FORMAT.format(
        post_title=safe_title,
        author_name=safe_author,
        content=safe_content,
        tags_html=safe_tags_html,
    )
