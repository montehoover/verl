from html import escape

def create_user_profile_html(user_name: str, user_bio: str) -> str:
    return f"<div><h2>{user_name}</h2><p>{user_bio}</p></div>"

def generate_tags_html(tags: list[str]) -> str:
    return " ".join(f"<span>{escape(tag, quote=True)}</span>" for tag in tags)
