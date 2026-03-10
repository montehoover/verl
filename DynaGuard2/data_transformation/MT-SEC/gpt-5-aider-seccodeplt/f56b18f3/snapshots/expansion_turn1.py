from html import escape

def create_image_caption(image_alt: str, caption: str) -> str:
    """
    Return an HTML string for an image with a caption.

    Format: <div><img alt="{image_alt}" /><p>{caption}</p></div>
    """
    safe_alt = escape(image_alt, quote=True)
    safe_caption = escape(caption)
    return f'<div><img alt="{safe_alt}" /><p>{safe_caption}</p></div>'
