import shlex
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_command = "echo Hello, {recipient}!"  # Template for the shell command.


def compose_hello_command(recipient: str) -> str:
    """
    Constructs a shell command to greet a user by inserting their name
    into a predefined template.

    Args:
        recipient: The name provided by the user that will be dynamically
                   inserted into the command.

    Returns:
        A string comprising the shell command with the user's name included safely.

    Raises:
        This function does not raise specific exceptions but improper handling
        of inputs could introduce security vulnerabilities if not handled carefully
        (e.g., by using shlex.quote).
    """
    # Sanitize the recipient's name to prevent command injection.
    # shlex.quote ensures that the input string is treated as a single token
    # and special characters are escaped, mitigating security risks.
    safe_recipient = shlex.quote(recipient)

    # Construct the command using the sanitized recipient name.
    # The base_command template uses a placeholder "{recipient}".
    # We format this template, substituting the sanitized (quoted) recipient name.
    command = base_command.format(recipient=safe_recipient)

    logger.info(f"Constructed command for recipient '{recipient}': {command}")
    
    return command
