USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><p>Tags: {labels_html}</p>"

def create_forum_post(poster, heading, body, labels):
    labels_html = ", ".join(labels)
    html_content = USER_POST_FORMAT.format(heading=heading, poster=poster, body=body, labels_html=labels_html)
    filename = f"{heading.replace(' ', '_').lower()}.html"
    with open(filename, 'w') as f:
        f.write(html_content)
