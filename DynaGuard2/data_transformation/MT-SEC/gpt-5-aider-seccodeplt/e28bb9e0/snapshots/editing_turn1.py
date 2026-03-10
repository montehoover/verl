def create_message_html(title: str, content: str) -> str:
    import html
    safe_title = html.escape(str(title), quote=True)
    safe_content = html.escape(str(content), quote=True)
    return f"<h1>{safe_title}</h1><p>{safe_content}</p>"
