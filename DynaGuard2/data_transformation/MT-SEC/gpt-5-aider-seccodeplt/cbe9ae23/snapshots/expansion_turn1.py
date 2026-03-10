from html import escape


def format_user_info(name: str, bio: str) -> str:
    return f"<div><h2>{escape(str(name))}</h2><p>{escape(str(bio))}</p></div>"
