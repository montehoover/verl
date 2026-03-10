def generate_user_content_html(author_name: str, content: str) -> str:
    return f"<div><h3>Posted by {author_name}</h3><p>{content}</p></div>"

def format_tags(tags: list[str]) -> str:
    return " ".join(f"<span>{tag}</span>" for tag in tags)
