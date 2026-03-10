from html import escape

BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"

def create_image_caption(image_alt: str, caption: str) -> str:
    """
    Return an HTML string for an image with a caption.

    Format: <div><img alt="{image_alt}" /><p>{caption}</p></div>
    """
    safe_alt = escape(image_alt, quote=True)
    safe_caption = escape(caption)
    return f'<div><img alt="{safe_alt}" /><p>{safe_caption}</p></div>'

def compose_bio_section(name: str, bio: str) -> str:
    """
    Return an HTML snippet for a photographer bio section.

    Format: <section><h2>{name}</h2><p>{bio}</p></section>
    Applies basic HTML escaping to avoid layout issues.
    """
    safe_name = escape(name, quote=True)
    safe_bio = escape(bio)
    return f'<section><h2>{safe_name}</h2><p>{safe_bio}</p></section>'

def create_blog_post(heading: str, text: str) -> str:
    """
    Return an HTML string for a blog post using BLOG_FORMAT.

    BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"
    """
    safe_heading = escape(heading, quote=True)
    safe_text = escape(text)
    return BLOG_FORMAT.format(heading=safe_heading, text=safe_text)
