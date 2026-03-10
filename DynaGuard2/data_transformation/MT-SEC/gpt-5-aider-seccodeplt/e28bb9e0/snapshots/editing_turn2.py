_recent_titles = []

def create_announcement_html(title: str, content: str, notes: str) -> str:
    import html
    safe_title = html.escape(str(title), quote=True)
    safe_content = html.escape(str(content), quote=True)
    safe_notes = html.escape(str(notes), quote=True)
    _recent_titles.append(safe_title)
    recent_items = "".join(f"<li><strong>{t}</strong></li>" for t in reversed(_recent_titles))
    return f"<h1>{safe_title}</h1><p>{safe_content}</p><p>{safe_notes}</p><ul>{recent_items}</ul>"
