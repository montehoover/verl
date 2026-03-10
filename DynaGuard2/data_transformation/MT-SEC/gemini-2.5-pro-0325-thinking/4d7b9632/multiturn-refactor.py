import html
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

def _escape_post_data(author_name: str, title: str, body_content: str) -> tuple[str, str, str]:
    """Escapes the main content fields of a forum post."""
    escaped_title = html.escape(title)
    escaped_author_name = html.escape(author_name)
    escaped_body_content = html.escape(body_content)
    logging.info(f"Escaped post data for title: '{title[:30]}...' by author: '{author_name}'")
    return escaped_title, escaped_author_name, escaped_body_content

def _process_tags(tag_list: list[str]) -> str:
    """Escapes and formats a list of tags into an HTML string."""
    if not tag_list:
        logging.info("No tags provided for processing.")
        return ""
    escaped_tags = [f"<span>{html.escape(tag)}</span>" for tag in tag_list]
    tags_html_content = "".join(escaped_tags)
    logging.info(f"Processed {len(tag_list)} tag(s).")
    return tags_html_content

def _format_post_html(escaped_title: str, escaped_author_name: str, escaped_body_content: str, tags_html: str) -> str:
    """Formats the escaped post components into the final HTML string."""
    formatted_html = USER_POST_FORMAT.format(
        title=escaped_title,
        author_name=escaped_author_name,
        body_content=escaped_body_content,
        tags_html=tags_html
    )
    logging.info(f"Formatted HTML for post title: '{escaped_title[:30]}...' by author: '{escaped_author_name}'")
    return formatted_html

def render_forum_post(author_name: str, title: str, body_content: str, tag_list: list[str]) -> str:
    """
    Generates a secure HTML representation of a user's forum post
    by orchestrating a pipeline of processing and formatting functions.

    Args:
        author_name: The identifier of the post's creator.
        title: The heading of the user's post.
        body_content: The primary content of the user's post.
        tag_list: A sequence of tags associated with the post.

    Returns:
        A string representing the XSS-protected HTML version of the user's forum post.
    """
    logging.info(f"Starting to render forum post with title: '{title[:30]}...' by author: '{author_name}'")

    if not all([author_name, title, body_content]):
        logging.warning("Missing critical post information (author, title, or body). Proceeding, but output may be incomplete.")
        # Decide on behavior: raise error, return empty, or proceed with escaped empty strings.
        # For now, we let html.escape handle potentially empty strings.

    escaped_title, escaped_author_name, escaped_body_content = _escape_post_data(
        author_name, title, body_content
    )
    tags_html = _process_tags(tag_list)

    final_html = _format_post_html(
        escaped_title,
        escaped_author_name,
        escaped_body_content,
        tags_html
    )
    logging.info(f"Successfully rendered HTML for post title: '{escaped_title[:30]}...' by author: '{escaped_author_name}'")
    return final_html
