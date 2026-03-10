import html


def forum_post_with_points(title: str, content: str, points: list[str]) -> None:
    """
    Print a simple HTML document representing a forum post, and append a plain text
    list of discussion points at the end.

    Args:
        title: The post title as a string.
        content: The post content as a string.
        points: A list of discussion point strings.
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

    # Create plain text list of points
    plain_points = ""
    if points:
        lines = [f"- {str(p)}" for p in points]
        plain_points = "Discussion points:\n" + "\n".join(lines)

    output = html_doc if not plain_points else f"{html_doc}\n\n{plain_points}"
    print(output)
