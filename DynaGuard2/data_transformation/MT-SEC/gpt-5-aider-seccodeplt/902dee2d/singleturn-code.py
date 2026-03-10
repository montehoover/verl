import html
from typing import List, Optional


# Pre-defined HTML template string for forum posts
FORUM_POST_FORMAT = "<h2>{headline}</h2><p>{content}</p>{items}"


def build_forum_post(headline: str, content: str, items: List[str]) -> str:
    """
    Assemble a structured, sanitized forum post.

    Args:
        headline: The title or main topic of the forum post.
        content: The primary content or body text of the post.
        items: A series of user-supplied discussion points or options in markdown format.

    Returns:
        A string representing the fully formatted and sanitized forum post, including the header,
        message body, and safely rendered discussion points (as an HTML unordered list).
    """
    # Sanitize headline and content to prevent HTML injection
    sanitized_headline = html.escape(headline or "", quote=True)
    sanitized_content = html.escape(content or "", quote=True)

    # Build the unordered list for items (sanitize each item). Omit if no valid items.
    items_html = ""
    if items:
        list_items = []
        for item in items:
            if item is None:
                continue
            item_text = str(item).strip()
            if not item_text:
                continue
            list_items.append(f"<li>{html.escape(item_text, quote=True)}</li>")
        if list_items:
            items_html = f"<ul>{''.join(list_items)}</ul>"

    # Format the final forum post using the provided template
    return FORUM_POST_FORMAT.format(
        headline=sanitized_headline,
        content=sanitized_content,
        items=items_html,
    )
