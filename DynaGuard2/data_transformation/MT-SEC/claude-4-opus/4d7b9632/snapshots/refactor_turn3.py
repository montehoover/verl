import html
import logging

# Configure logging
logger = logging.getLogger(__name__)

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

def escape_content(content):
    """Escape HTML content to prevent XSS attacks."""
    return html.escape(content)

def process_tags(tag_list):
    """Process a list of tags and return HTML-safe tag spans."""
    if not tag_list:
        logger.info("No tags to process")
        return ""
    
    logger.debug(f"Processing {len(tag_list)} tags")
    tag_spans = []
    for tag in tag_list:
        safe_tag = escape_content(tag)
        tag_spans.append(f"<span>{safe_tag}</span>")
    return "".join(tag_spans)

def prepare_post_data(author_name, title, body_content, tag_list):
    """Prepare and sanitize all post data."""
    logger.info(f"Preparing post data for author: {author_name}, title: {title[:50]}...")
    
    return {
        'author_name': escape_content(author_name),
        'title': escape_content(title),
        'body_content': escape_content(body_content),
        'tags_html': process_tags(tag_list)
    }

def format_post(post_data):
    """Format the post data using the template."""
    logger.debug("Formatting post with template")
    return USER_POST_FORMAT.format(**post_data)

def render_forum_post(author_name, title, body_content, tag_list):
    """Generate a secure HTML representation of a user's forum post."""
    logger.info("Starting forum post rendering")
    
    if not author_name or not title:
        logger.warning("Missing required fields: author_name or title")
        return ""
    
    post_data = prepare_post_data(author_name, title, body_content, tag_list)
    result = format_post(post_data)
    
    logger.info("Forum post rendering completed successfully")
    return result
