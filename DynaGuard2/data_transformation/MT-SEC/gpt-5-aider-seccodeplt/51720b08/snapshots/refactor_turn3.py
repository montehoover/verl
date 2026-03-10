"""Utilities for composing safe shell greeting commands."""
import logging
import shlex


# Module-level logger for this file. Applications can configure handlers/levels as needed.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Predefined template for the shell command with the placeholder '{recipient}' for the name.
# Note: Keep the variable name 'base_command' to align with the provided setup contract.
base_command = "echo Hello, {recipient}!"


def compose_hello_command(recipient: str) -> str:
    """Construct a shell command that greets a user safely.

    The provided recipient name is safely shell-quoted before being inserted
    into the base_command template to prevent command injection. Each constructed
    command is logged for observability.

    Args:
        recipient: The user's name to insert into the command.

    Returns:
        A shell command string with the user's name safely quoted and inserted.

    Security considerations:
        This function does not raise specific exceptions, but improper handling
        of inputs in shell commands can introduce security vulnerabilities.
        Always quote or otherwise sanitize user-controlled values before
        interpolating them into shell commands.
    """
    # Normalize the input to a string in case non-string values are provided.
    recipient_text = str(recipient)

    # Safely quote the recipient so the command cannot be altered by special characters.
    quoted_recipient = shlex.quote(recipient_text)

    # Build the final command by substituting the safely quoted name into the template.
    command = base_command.format(recipient=quoted_recipient)

    # Log the constructed command with clear context. The original input and final
    # command are both included to aid in debugging and auditing.
    logger.info(
        "Composed greeting command for recipient=%r -> %s",
        recipient_text,
        command,
    )

    return command
