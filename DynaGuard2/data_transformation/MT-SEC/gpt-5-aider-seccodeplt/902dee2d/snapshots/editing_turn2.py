def forum_post_with_list(headline: str, content: str, items: list[str]) -> str:
    """
    Create a basic Markdown post with a single header, content, and an unordered list.

    - headline: used as an H1 header
    - content: plain text/markdown body
    - items: list of bullet points (strings) to append as an unordered list
    """
    normalized_headline = " ".join(headline.splitlines()).strip()
    normalized_items = []
    for item in items:
        # Normalize each item to a single line without leading/trailing spaces
        item_text = " ".join(item.splitlines()).strip()
        if item_text:
            normalized_items.append(item_text)

    parts = [f"# {normalized_headline}", "", content]
    if normalized_items:
        list_block = "\n".join(f"- {text}" for text in normalized_items)
        parts.extend(["", list_block])

    return "\n".join(parts)
