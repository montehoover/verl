FORUM_POST_FORMAT = "<h2>{headline}</h2><p>{content}</p>{items}"

def build_forum_post(headline: str, content: str, items: list[str]) -> str:
    # Convert the list of items to HTML unordered list
    items_html = "<ul>"
    for item in items:
        items_html += f"<li>{item}</li>"
    items_html += "</ul>"
    
    # Format the forum post using the template
    return FORUM_POST_FORMAT.format(
        headline=headline,
        content=content,
        items=items_html
    )
