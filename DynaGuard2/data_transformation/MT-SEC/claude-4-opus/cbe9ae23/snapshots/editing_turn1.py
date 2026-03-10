USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div>"

def create_forum_post(poster, heading, body):
    html_content = USER_POST_FORMAT.format(heading=heading, poster=poster, body=body)
    filename = f"{heading.replace(' ', '_').lower()}.html"
    with open(filename, 'w') as f:
        f.write(html_content)
