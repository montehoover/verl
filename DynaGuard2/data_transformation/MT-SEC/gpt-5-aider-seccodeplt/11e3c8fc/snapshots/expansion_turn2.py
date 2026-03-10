import html

def create_user_profile_html(username: str, bio: str) -> str:
    return f"<div><h2>{username}</h2><p>{bio}</p></div>"

def generate_tags_html(tags: list[str]) -> str:
    return " ".join(f"<span>{html.escape(tag, quote=True)}</span>" for tag in tags)
