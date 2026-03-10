import html

BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"

def publish_blog_article(subject: str, description: str) -> str:
    sanitized_subject = html.escape(subject)
    sanitized_description = html.escape(description)
    return BLOG_FORMAT.format(subject=sanitized_subject, description=sanitized_description)
