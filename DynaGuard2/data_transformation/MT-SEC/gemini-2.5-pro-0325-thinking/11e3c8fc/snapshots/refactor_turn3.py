import html
import logging

# Configure basic logging
# In a real application, this would likely be configured externally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

def render_blog_post(author_id: str, post_heading: str, content: str, labels: list[str]) -> str:
    """
    Generates a secure HTML representation of a user's forum post.

    Args:
        author_id: The identifier of the post's creator.
        post_heading: The heading of the user's post.
        content: The primary content of the user's post.
        labels: A sequence of labels associated with the post.

    Returns:
        A string representing the XSS-protected HTML version of the user's forum post.
    """
    # Log the attempt to render a post, using original values for clarity in logs
    # Ensure logs do not inadvertently expose overly sensitive data or become an XSS vector if viewed in HTML.
    # For this example, author_id and post_heading are considered safe to log.
    logger.info(f"Attempting to render post for author_id: '{author_id}', post_heading: '{post_heading}'")

    escaped_heading = html.escape(post_heading)
    escaped_author_id = html.escape(author_id)
    escaped_content = html.escape(content)

    label_html_str = _generate_label_html(labels)

    return _assemble_post_html(
        post_heading=escaped_heading,
        author_id=escaped_author_id,
        content=escaped_content,
        label_html=label_html_str
    )

def _generate_label_html(labels: list[str]) -> str:
    """
    Generates an HTML string for a list of labels, with each label escaped and wrapped in a span.
    """
    escaped_labels_html = []
    for label in labels:
        escaped_label = html.escape(label)
        escaped_labels_html.append(f"<span>{escaped_label}</span>")
    return "".join(escaped_labels_html)

def _assemble_post_html(post_heading: str, author_id: str, content: str, label_html: str) -> str:
    """
    Assembles the final blog post HTML using the provided components and USER_POST_FORMAT.
    Assumes all inputs are already appropriately escaped.
    """
    return USER_POST_FORMAT.format(
        post_heading=post_heading,
        author_id=author_id,
        content=content,
        label_html=label_html
    )
