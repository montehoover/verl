BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"

def html_format_message(title, content):
    return f"<h1>{title}</h1><p>{content}</p>"

def display_blog_entry(headline, text):
    return BLOG_FORMAT.format(headline=headline, text=text)
