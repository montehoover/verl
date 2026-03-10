import subprocess
import shlex
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_command = "echo Hello, {user_name}!"

def sanitize_user_input(user_name: str) -> str:
    """Sanitize user input to prevent command injection."""
    return shlex.quote(user_name)

def create_greeting_command(sanitized_name: str) -> str:
    """Create the greeting command with sanitized input."""
    return base_command.format(user_name=sanitized_name)

def generate_greeting_command(user_name: str) -> str:
    logger.info(f"Generating greeting command for user: {user_name}")
    sanitized_name = sanitize_user_input(user_name)
    command = create_greeting_command(sanitized_name)
    logger.info(f"Generated command: {command}")
    return command
