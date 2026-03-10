import html
import logging

# Configure logging
logger = logging.getLogger(__name__)

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"


def generate_tags_html(tag_list):
    """
    Generate HTML for tags, with each tag in a separate span element.
    
    Args:
        tag_list (list): A list of tag strings to be converted to HTML.
        
    Returns:
        str: HTML string with each tag wrapped in a span element and properly escaped.
    """
    return ''.join(f"<span>{html.escape(tag)}</span>" for tag in tag_list)


def sanitize_inputs(writer_name, title, body):
    """
    Sanitize all text inputs to prevent XSS attacks.
    
    Args:
        writer_name (str): The name of the post's author.
        title (str): The headline of the user's post.
        body (str): The primary text content of the user's post.
        
    Returns:
        dict: A dictionary containing sanitized versions of all input strings.
    """
    return {
        'writer_name': html.escape(writer_name),
        'title': html.escape(title),
        'body': html.escape(body)
    }


def generate_post_html(writer_name, title, body, tag_list):
    """
    Construct an HTML representation of a user's forum post.
    
    This function takes the components of a forum post and generates a safe HTML
    representation with XSS protection. All user inputs are sanitized before being
    inserted into the HTML template.
    
    Args:
        writer_name (str): The name of the post's author.
        title (str): The headline of the user's post.
        body (str): The primary text content of the user's post.
        tag_list (list): A collection of tags related to the post.
        
    Returns:
        str: XSS-protected HTML representation of the user's forum post.
    """
    # Log the input arguments
    logger.debug(
        "Generating post HTML with inputs - writer_name: %s, title: %s, "
        "body: %s, tags: %s",
        writer_name, title, body, tag_list
    )
    
    # Step 1: Sanitize all text inputs
    sanitized = sanitize_inputs(writer_name, title, body)
    
    # Step 2: Generate tags HTML
    tags_html = generate_tags_html(tag_list)
    
    # Step 3: Format the post using the template
    post_html = USER_POST_FORMAT.format(
        title=sanitized['title'],
        writer_name=sanitized['writer_name'],
        body=sanitized['body'],
        tags_html=tags_html
    )
    
    # Log the generated HTML output
    logger.debug("Generated HTML output: %s", post_html)
    
    return post_html
