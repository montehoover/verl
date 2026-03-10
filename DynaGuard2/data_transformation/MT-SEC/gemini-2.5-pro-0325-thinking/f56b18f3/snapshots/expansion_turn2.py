import html


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
