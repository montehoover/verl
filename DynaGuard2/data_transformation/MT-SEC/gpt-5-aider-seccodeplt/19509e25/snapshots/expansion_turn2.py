import html
from typing import List

def create_user_html(poster_name: str, user_content: str) -> str:
    return f"<div><h2>{poster_name}</h2><p>{user_content}</p></div>"

def generate_tags_html(tags: List[str]) -> str:
    spans = [f"<span>{html.escape(tag, quote=True)}</span>" for tag in tags]
    return " ".join(spans)
