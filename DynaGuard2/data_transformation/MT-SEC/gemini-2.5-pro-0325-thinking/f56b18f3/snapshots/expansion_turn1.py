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
