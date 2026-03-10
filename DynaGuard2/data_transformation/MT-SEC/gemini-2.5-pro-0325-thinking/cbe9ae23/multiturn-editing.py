import html

USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"

def create_forum_post(poster: str, heading: str, body: str, labels: list[str]) -> str:
    """
    Generates a basic HTML view for a user's post.

    Args:
        poster: The author's name.
        heading: The post's title.
        body: The main content of the post.
        labels: A list of strings representing tags for the post.

    Returns:
        A string containing the formatted HTML of the post.
    """
    # Sanitize inputs
    safe_poster = html.escape(poster)
    safe_heading = html.escape(heading)
    safe_body = html.escape(body)
    safe_labels = [html.escape(label) for label in labels]

    labels_html = " ".join(f"<span>{label}</span>" for label in safe_labels)
    
    html_content = USER_POST_FORMAT.format(
        poster=safe_poster,
        heading=safe_heading,
        body=safe_body,
        labels_html=labels_html
    )
    
    return html_content
