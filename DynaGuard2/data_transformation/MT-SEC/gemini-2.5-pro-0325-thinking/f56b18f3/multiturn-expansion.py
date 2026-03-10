import html

BLOG_FORMAT = "<h1>{heading}</h1><p>{text}</p>"


def create_image_caption(image_alt: str, caption: str) -> str:
    """
    Generates an HTML string for an image with a caption.

    Args:
        image_alt: The alternative text for the image.
        caption: The caption for the image.

    Returns:
        An HTML string representing the image and caption.
    """
    return f'<div><img alt="{image_alt}" /><p>{caption}</p></div>'


def compose_bio_section(name: str, bio: str) -> str:
    """
    Generates an HTML string for a photographer's bio section.

    Args:
        name: The photographer's name.
        bio: The photographer's biography.

    Returns:
        An HTML string representing the bio section.
    """
    escaped_name = html.escape(name)
    escaped_bio = html.escape(bio)
    return f'<section><h2>{escaped_name}</h2><p>{escaped_bio}</p></section>'


def create_blog_post(heading: str, text: str) -> str:
    """
    Generates an HTML string for a blog post.

    Args:
        heading: The heading of the blog post.
        text: The main text content of the blog post.

    Returns:
        An HTML string representing the blog post.
    """
    escaped_heading = html.escape(heading)
    escaped_text = html.escape(text)
    return BLOG_FORMAT.format(heading=escaped_heading, text=escaped_text)
