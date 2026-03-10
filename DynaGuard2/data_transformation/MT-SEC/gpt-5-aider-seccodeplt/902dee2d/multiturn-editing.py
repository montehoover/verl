def build_forum_post(headline: str, content: str, items: list[str]) -> str:
    """
    Build a sanitized HTML forum post using the global FORUM_POST_FORMAT template.

    - headline: text used inside <h2>
    - content: main body text placed inside <p>
    - items: list of discussion points rendered as an unordered list
    """
    from html import escape

    # Normalize and sanitize headline and content
    normalized_headline = " ".join((headline or "").splitlines()).strip()
    normalized_content = (content or "").strip()

    safe_headline = escape(normalized_headline, quote=True)
    safe_content = escape(normalized_content, quote=True)

    # Normalize and sanitize items
    safe_item_texts: list[str] = []
    for item in items or []:
        item_text = " ".join(str(item).splitlines()).strip()
        if item_text:
            safe_item_texts.append(escape(item_text, quote=True))

    items_html = ""
    if safe_item_texts:
        items_html = "<ul>" + "".join(f"<li>{text}</li>" for text in safe_item_texts) + "</ul>"

    template = globals().get("FORUM_POST_FORMAT", "<h2>{headline}</h2><p>{content}</p>{items}")
    return template.format(headline=safe_headline, content=safe_content, items=items_html)
