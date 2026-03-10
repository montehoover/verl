import html

USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"


def create_labels_html(labels):
    """Create HTML representation of labels as span elements."""
    return ''.join([f'<span>{html.escape(label)}</span>' for label in labels])


def format_post(poster, heading, body, labels_html):
    """Fill in the USER_POST_FORMAT template with sanitized values."""
    return USER_POST_FORMAT.format(
        poster=html.escape(poster),
        heading=html.escape(heading),
        body=html.escape(body),
        labels_html=labels_html
    )


def create_forum_post(poster, heading, body, labels):
    """
    Generate an HTML representation of a user's post for a public forum.
    
    Args:
        poster (str): The author of the post.
        heading (str): The title of the user's post.
        body (str): The main content of the user's post.
        labels (list[str]): A list of tags associated with the post.
    
    Returns:
        str: A string containing the sanitized HTML representation of the user's post.
    """
    # Create label HTML
    labels_html = create_labels_html(labels)
    
    # Format and return the post
    return format_post(poster, heading, body, labels_html)
