def basic_forum_post(post_title: str, post_body: str) -> str:
    def _escape_xml(text: str) -> str:
        # Escape special XML characters: &, <, >, ", '
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    title = "" if post_title is None else str(post_title)
    body = "" if post_body is None else str(post_body)

    return f"<title>{_escape_xml(title)}</title><content>{_escape_xml(body)}</content>"
