def format_blog_post(title, content):
    return f"# {title}\n{content}"

BLOG_FORMAT = "<h1>{subject}</h1><p>{description}</p>"

def publish_blog_article(subject, description):
    return BLOG_FORMAT.format(subject=subject, description=description)
