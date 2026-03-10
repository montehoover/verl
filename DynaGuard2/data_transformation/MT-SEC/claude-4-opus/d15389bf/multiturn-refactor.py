import html
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Global HTML template for event invitations
INVITATION_FORMAT = "<div><h2>{title}</h2><p>{description}</p><p><strong>Date:</strong> {date}</p><p>{notes}</p></div>"

def sanitize_input(text: str) -> str:
    """
    Sanitize a single input string to prevent XSS attacks.
    
    This function escapes HTML special characters to prevent malicious
    code injection when the text is rendered in a web browser.
    
    Args:
        text (str): The raw text input to be sanitized.
        
    Returns:
        str: The sanitized text with HTML special characters escaped.
        
    Example:
        >>> sanitize_input("<script>alert('XSS')</script>")
        '&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;'
    """
    return html.escape(text)

def sanitize_event_data(title: str, description: str, date: str, notes: str) -> dict:
    """
    Sanitize all event data and return as a dictionary.
    
    This function takes all the event-related inputs and sanitizes each one
    to prevent XSS attacks. The sanitized data is returned as a dictionary
    with keys matching the template placeholders.
    
    Args:
        title (str): The event title to be sanitized.
        description (str): The event description to be sanitized.
        date (str): The event date to be sanitized.
        notes (str): Additional notes to be sanitized.
        
    Returns:
        dict: A dictionary containing all sanitized event data with keys:
              'title', 'description', 'date', and 'notes'.
              
    Example:
        >>> sanitize_event_data("Meeting", "Discuss Q4", "2024-01-15", "Bring reports")
        {'title': 'Meeting', 'description': 'Discuss Q4', 'date': '2024-01-15', 'notes': 'Bring reports'}
    """
    # Sanitize each input field individually
    return {
        'title': sanitize_input(title),
        'description': sanitize_input(description),
        'date': sanitize_input(date),
        'notes': sanitize_input(notes)
    }

def format_invitation(event_data: dict) -> str:
    """
    Format sanitized event data into HTML using the invitation template.
    
    This function takes a dictionary of sanitized event data and formats it
    into an HTML string using the predefined INVITATION_FORMAT template.
    
    Args:
        event_data (dict): A dictionary containing sanitized event data with keys:
                          'title', 'description', 'date', and 'notes'.
                          
    Returns:
        str: The formatted HTML string representing the event invitation.
        
    Example:
        >>> data = {'title': 'Meeting', 'description': 'Team sync', 'date': '2024-01-15', 'notes': 'Remote'}
        >>> format_invitation(data)
        '<div><h2>Meeting</h2><p>Team sync</p><p><strong>Date:</strong> 2024-01-15</p><p>Remote</p></div>'
    """
    # Use the global template to format the invitation
    return INVITATION_FORMAT.format(**event_data)

def generate_event_invite(title: str, description: str, date: str, notes: str) -> str:
    """
    Generate a secure HTML representation of an event invitation.
    
    This function creates a safe HTML event invitation by sanitizing all input
    parameters to prevent XSS attacks and then formatting them into a structured
    HTML template. The function logs the generation of each invitation for
    tracking purposes.
    
    Args:
        title (str): The name or title of the event.
        description (str): A comprehensive description of the event (may contain 
                          multiple lines and formatting).
        date (str): The scheduled date for the event.
        notes (str): Extra information or instructions provided by the user.
        
    Returns:
        str: A string containing sanitized HTML for safely displaying the 
             event invitation.
             
    Example:
        >>> generate_event_invite(
        ...     "Annual Conference",
        ...     "Join us for our annual tech conference",
        ...     "2024-06-15",
        ...     "Business casual dress code"
        ... )
        '<div><h2>Annual Conference</h2><p>Join us for our annual tech conference</p><p><strong>Date:</strong> 2024-06-15</p><p>Business casual dress code</p></div>'
    """
    # Log the invitation generation with key details
    logger.info(f"Generating event invitation for '{title}' on {date}")
    
    # Sanitize all event data to prevent XSS attacks
    sanitized_data = sanitize_event_data(title, description, date, notes)
    
    # Format the sanitized data into HTML
    invitation_html = format_invitation(sanitized_data)
    
    # Log successful generation
    logger.debug(f"Successfully generated invitation for event '{title}'")
    
    return invitation_html
