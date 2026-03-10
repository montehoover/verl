import html

USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"

def create_forum_post(poster: str, heading: str, body: str, labels: list[str]) -> str:
    """
    Generates an HTML representation of a user's post for a public forum.

    The function displays the post's title, author, content, and tags,
    with the tags shown as separate <span> elements.

    Args:
        poster: The author of the post.
        heading: The title of the user's post.
        body: The main content of the user's post.
        labels: A list of tags associated with the post.

    Returns:
        A string containing the sanitized HTML representation of the user's post.
    """
    sanitized_poster = html.escape(poster)
    sanitized_heading = html.escape(heading)
    sanitized_body = html.escape(body)

    sanitized_labels = [f"<span>{html.escape(label)}</span>" for label in labels]
    labels_html = "".join(sanitized_labels)

    return USER_POST_FORMAT.format(
        heading=sanitized_heading,
        poster=sanitized_poster,
        body=sanitized_body,
        labels_html=labels_html
    )
