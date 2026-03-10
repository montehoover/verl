import html

USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"

def create_forum_post(poster, heading, body, labels):
    # Sanitize inputs
    poster = html.escape(poster)
    heading = html.escape(heading)
    body = html.escape(body)
    
    # Create label spans
    labels_html = ''.join([f'<span>{html.escape(label)}</span>' for label in labels])
    
    # Format and return the post
    return USER_POST_FORMAT.format(
        poster=poster,
        heading=heading,
        body=body,
        labels_html=labels_html
    )
