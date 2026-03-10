def generate_post_html(title: str, content: str) -> str:
    return f"<div><h2>{title}</h2><p>{content}</p></div>"

def format_options_html(options: list[str]) -> str:
    items = ''.join(f"<li>{option}</li>" for option in options)
    return f"<ul>{items}</ul>"
