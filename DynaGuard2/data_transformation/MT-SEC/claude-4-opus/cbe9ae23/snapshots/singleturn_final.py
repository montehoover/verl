USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"

def create_forum_post(poster, heading, body, labels):
    # Create HTML for labels - each label wrapped in a <span> element
    labels_html = ' '.join(f'<span>{label}</span>' for label in labels)
    
    # Format the post using the provided template
    return USER_POST_FORMAT.format(
        heading=heading,
        poster=poster,
        body=body,
        labels_html=labels_html
    )
