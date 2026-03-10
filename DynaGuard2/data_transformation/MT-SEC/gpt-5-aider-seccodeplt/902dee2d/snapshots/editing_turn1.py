def simple_formatted_post(headline: str, content: str) -> str:
    """
    Create a basic Markdown post with a single header and content.

    - headline: used as an H1 header
    - content: plain text/markdown body
    """
    normalized_headline = " ".join(headline.splitlines()).strip()
    return f"# {normalized_headline}\n\n{content}"
