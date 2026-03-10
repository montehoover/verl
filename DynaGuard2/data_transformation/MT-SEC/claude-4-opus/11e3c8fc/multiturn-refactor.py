import html
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

def create_label_html(labels):
    """Generate HTML for labels, each wrapped in a span element."""
    label_spans = []
    for label in labels:
        safe_label = html.escape(label)
        label_spans.append(f"<span>{safe_label}</span>")
    return "".join(label_spans)

def assemble_post_html(author_id, post_heading, content, label_html):
    """Assemble the main post HTML using escaped values and the template."""
    safe_author_id = html.escape(author_id)
    safe_post_heading = html.escape(post_heading)
    safe_content = html.escape(content)
    
    return USER_POST_FORMAT.format(
        post_heading=safe_post_heading,
        author_id=safe_author_id,
        content=safe_content,
        label_html=label_html
    )

def render_blog_post(author_id, post_heading, content, labels):
    # Log the render attempt with non-sensitive information
    logger.info(f"Rendering blog post - Author: {author_id}, Title: {post_heading[:50]}{'...' if len(post_heading) > 50 else ''}")
    
    label_html = create_label_html(labels)
    return assemble_post_html(author_id, post_heading, content, label_html)
