def create_user_html(author_name: str, user_bio: str) -> str:
    """
    Return an HTML string with the author's name and bio wrapped in a div.
    Format: <div><h2>{author_name}</h2><p>{user_bio}</p></div>
    """
    return f"<div><h2>{author_name}</h2><p>{user_bio}</p></div>"
