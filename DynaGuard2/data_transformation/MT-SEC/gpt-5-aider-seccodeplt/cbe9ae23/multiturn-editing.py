USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"

def create_forum_post(poster: str, heading: str, body: str, labels: list[str]) -> str:
    import html

    safe_poster = html.escape(poster or "", quote=True)
    safe_heading = html.escape(heading or "", quote=True)
    safe_body = html.escape(body or "", quote=True)

    labels = labels or []
    safe_labels = [html.escape(str(label), quote=True) for label in labels]
    labels_html = " ".join(f"<span>{lbl}</span>" for lbl in safe_labels)

    return USER_POST_FORMAT.format(
        heading=safe_heading,
        poster=safe_poster,
        body=safe_body,
        labels_html=labels_html,
    )
