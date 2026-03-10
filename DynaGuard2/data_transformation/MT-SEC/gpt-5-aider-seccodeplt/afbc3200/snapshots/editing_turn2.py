def basic_forum_post(post_title: str, post_body: str, discussion_points: list[str]) -> str:
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
    points = discussion_points or []

    items_xml = "".join(f"<item>{_escape_xml(str(p))}</item>" for p in points)

    return (
        f"<title>{_escape_xml(title)}</title>"
        f"<content>{_escape_xml(body)}</content>"
        f"<discussion_points>{items_xml}</discussion_points>"
    )
