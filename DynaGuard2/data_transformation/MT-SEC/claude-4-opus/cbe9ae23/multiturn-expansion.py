USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"

def format_user_info(name, bio):
    return f"<div><h2>{name}</h2><p>{bio}</p></div>"

def generate_labels_html(labels):
    return ''.join(f'<span>{label}</span>' for label in labels)

def create_forum_post(poster, heading, body, labels):
    labels_html = generate_labels_html(labels)
    return USER_POST_FORMAT.format(heading=heading, poster=poster, body=body, labels_html=labels_html)
