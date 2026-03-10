import html


def basic_forum_post(title: str, content: str) -> None:
    """
    Print a simple HTML document representing a forum post.

    Args:
        title: The post title as a string.
        content: The post content as a string.
    """
    safe_title = html.escape(title, quote=True)
    safe_content = html.escape(content, quote=True)

    html_doc = (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        f"  <title>{safe_title}</title>\n"
        "</head>\n"
        "<body>\n"
        f"  <h2>{safe_title}</h2>\n"
        f"  <p>{safe_content}</p>\n"
        "</body>\n"
        "</html>"
    )
    print(html_doc)
