from html import escape

USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div><div>{labels_html}</div>"


def format_user_info(name: str, bio: str) -> str:
    return f"<div><h2>{escape(str(name))}</h2><p>{escape(str(bio))}</p></div>"


def generate_labels_html(labels: list[str]) -> str:
    spans = []
    for label in labels:
        text = str(label).strip()
        if not text:
            continue
        spans.append(f"<span>{escape(text)}</span>")
    return "".join(spans)


def create_forum_post(poster: str, heading: str, body: str, labels: list[str]) -> str:
    safe_poster = escape(str(poster).strip())
    safe_heading = escape(str(heading).strip())
    safe_body = escape(str(body))
    labels_html = generate_labels_html(labels or [])
    return USER_POST_FORMAT.format(
        poster=safe_poster,
        heading=safe_heading,
        body=safe_body,
        labels_html=labels_html,
    )
