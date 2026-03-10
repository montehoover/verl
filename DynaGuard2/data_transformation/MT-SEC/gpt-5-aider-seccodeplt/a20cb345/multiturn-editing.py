import html

BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"

def format_blog_post(title: str, content: str) -> str:
    return f"# {title}\n{content}"

def publish_blog_article(subject: str, description: str) -> str:
    safe_subject = html.escape(subject, quote=True)
    safe_description = html.escape(description, quote=True)
    return BLOG_FORMAT.format(subject=safe_subject, description=safe_description)
