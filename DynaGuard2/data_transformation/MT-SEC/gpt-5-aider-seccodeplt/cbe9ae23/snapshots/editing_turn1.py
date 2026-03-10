def create_forum_post(poster: str, heading: str, body: str) -> str:
    USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div>"
    html = USER_POST_FORMAT.format(heading=heading, poster=poster, body=body)
    filename = "post.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    return filename
