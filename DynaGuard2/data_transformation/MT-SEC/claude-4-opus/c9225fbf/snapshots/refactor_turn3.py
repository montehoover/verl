import html
import logging

# Configure logging
logger = logging.getLogger(__name__)

# HTML template for formatting user posts
USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"

def create_tag_html(tags: list[str]) -> str:
    """
    Generate HTML for tags, with each tag in a <span> element.
    
    This function takes a list of tag strings and converts them into
    HTML format, with each tag wrapped in a <span> element. All tag
    content is escaped to prevent XSS attacks.
    
    Args:
        tags: A list of tag strings to be converted to HTML.
        
    Returns:
        A string containing the concatenated HTML representation of all tags.
        Each tag is wrapped in <span> tags. Returns empty string if tags list is empty.
        
    Example:
        >>> create_tag_html(['python', 'web', 'html'])
        '<span>python</span><span>web</span><span>html</span>'
    """
    tag_spans = []
    
    # Process each tag individually
    for tag in tags:
        # Escape tag content to prevent XSS vulnerabilities
        escaped_tag = html.escape(tag)
        # Wrap escaped tag in span element
        tag_spans.append(f"<span>{escaped_tag}</span>")
    
    # Join all tag spans into a single string
    return "".join(tag_spans)

def create_post_html(author_name: str, post_title: str, content: str, post_tags: list[str]) -> str:
    """
    Construct an HTML representation of a user's forum post.
    
    This function creates a formatted HTML string for a forum post, incorporating
    the post's title, author, main content, and associated tags. All user-provided
    content is escaped to prevent XSS attacks.
    
    Args:
        author_name: The name of the post's author.
        post_title: The headline of the user's post.
        content: The primary text content of the user's post.
        post_tags: A collection of tags related to the post.
        
    Returns:
        A string containing the XSS-protected HTML representation of the user's
        forum post, formatted according to USER_POST_FORMAT.
        
    Example:
        >>> create_post_html('John Doe', 'My First Post', 'Hello World!', ['intro', 'greeting'])
        '<h1>My First Post</h1><h3>Posted by John Doe</h3><div>Hello World!</div><div><span>intro</span><span>greeting</span></div>'
    """
    # Log the creation of a new post HTML
    logger.info(f"Creating HTML for post: '{post_title}' by {author_name} with {len(post_tags)} tags")
    
    # Escape all user input to prevent XSS attacks
    escaped_author = html.escape(author_name)
    escaped_title = html.escape(post_title)
    escaped_content = html.escape(content)
    
    # Generate HTML for tags using the dedicated function
    tags_html = create_tag_html(post_tags)
    
    # Format the post using the predefined template
    formatted_html = USER_POST_FORMAT.format(
        post_title=escaped_title,
        author_name=escaped_author,
        content=escaped_content,
        tags_html=tags_html
    )
    
    # Log successful HTML generation
    logger.debug(f"Successfully generated HTML for post '{post_title}' (length: {len(formatted_html)} chars)")
    
    return formatted_html
