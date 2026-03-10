if 'FORUM_POST_FORMAT' not in globals():
    FORUM_POST_FORMAT = "<h2>{header}</h2><p>{message}</p>{options}"

def build_forum_entry(post_title: str, post_body: str, bullet_points: list[str]) -> str:
    def _escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )

    title = "" if post_title is None else str(post_title)
    body = "" if post_body is None else str(post_body)
    points = bullet_points or []

    items_html = "".join(f"<li>{_escape_html(str(p))}</li>" for p in points)
    options_html = f"<ul>{items_html}</ul>"

    return FORUM_POST_FORMAT.format(
        header=_escape_html(title),
        message=_escape_html(body),
        options=options_html,
    )
