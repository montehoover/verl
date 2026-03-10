import html
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def _generate_tags_html(tag_list: list[str]) -> str:
    """
    Generates an HTML string for a list of tags.

    Each tag in the input list is HTML-escaped and wrapped in a <span> element.
    The resulting <span> elements are then concatenated into a single string.

    Args:
        tag_list: A list of strings, where each string is a tag.

    Returns:
        A string containing the HTML representation of the tags.
        Returns an empty string if the tag_list is empty.
    """
    escaped_tags = [f"<span>{html.escape(tag)}</span>" for tag in tag_list]
    return "".join(escaped_tags)

def generate_post_html(writer_name: str, title: str, body: str, tag_list: list[str]) -> str:
    """
    Constructs an HTML representation of a user's forum post, ensuring all
    user-supplied content is XSS-protected.

    The function takes the author's name, post title, body content, and a list
    of tags. It escapes these inputs to prevent XSS attacks, generates HTML
    for the tags using a helper function, and then formats the entire post
    using a predefined HTML template.

    Args:
        writer_name: The name of the post's author (str).
        title: The headline of the user's post (str).
        body: The primary text content of the user's post (str).
        tag_list: A list of strings representing tags related to the post (list[str]).

    Returns:
        A string containing the XSS-protected HTML representation of the user's
        forum post.
    """
    logger.info(
        f"Generating post HTML for writer: '{writer_name}', title: '{title}', "
        f"body_len: {len(body)}, tags: {tag_list}"
    )

    escaped_writer_name = html.escape(writer_name)
    escaped_title = html.escape(title)
    escaped_body = html.escape(body)

    # Generate HTML for tags using the helper function
    tags_html_content = _generate_tags_html(tag_list)

    # Format the final post HTML
    post_html = USER_POST_FORMAT.format(
        title=escaped_title,
        writer_name=escaped_writer_name,
        body=escaped_body,
        tags_html=tags_html_content
    )

    logger.info(f"Generated HTML (first 100 chars): {post_html[:100]}...")
    return post_html
