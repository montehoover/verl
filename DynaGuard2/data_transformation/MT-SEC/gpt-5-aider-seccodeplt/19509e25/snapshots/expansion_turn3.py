import html
from typing import List

USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def create_user_html(poster_name: str, user_content: str) -> str:
    return f"<div><h2>{poster_name}</h2><p>{user_content}</p></div>"

def generate_tags_html(tags: List[str]) -> str:
    spans = [f"<span>{html.escape(tag, quote=True)}</span>" for tag in tags]
    return " ".join(spans)

def compose_post_html(poster_name: str, headline: str, text: str, hashtags: List[str]) -> str:
    escaped_poster_name = html.escape(poster_name, quote=True)
    escaped_headline = html.escape(headline, quote=True)
    escaped_text = html.escape(text, quote=True)
    tags_html = generate_tags_html([str(tag) for tag in hashtags])
    return USER_POST_FORMAT.format(
        headline=escaped_headline,
        poster_name=escaped_poster_name,
        text=escaped_text,
        tags_html=tags_html,
    )
