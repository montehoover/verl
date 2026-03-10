import re
import shlex

def assemble_greet_command(name_of_user: str):
    """
    Generates a shell command to display a greeting message with a user's name.

    Args:
        name_of_user: The name of the user to include in the greeting.
                      Only letters, numbers, and spaces are allowed.

    Returns:
        A shell command string.

    Raises:
        ValueError: If name_of_user contains invalid characters.
    """
    base_command_template = "echo Hello, {name_placeholder}!"

    # Validate the name_of_user to prevent command injection.
    # Allow letters, numbers, and spaces.
    if not re.match(r"^[a-zA-Z0-9 ]*$", name_of_user):
        raise ValueError("name_of_user contains invalid characters. Only letters, numbers, and spaces are allowed.")

    # Safely quote the user input to be used within the echo command's string.
    # shlex.quote is generally good, but here we are inserting into a string that itself will be parsed by the shell.
    # The regex validation above is the primary defense for this specific template.
    # If the template were more complex, or characters like " were allowed, shlex.quote would be more critical.
    # For this specific template "echo Hello, {name}!", the regex is sufficient.
    
    # Replace the placeholder with the validated name.
    # Using f-string or .replace() after validation is fine.
    # The placeholder name in the template is changed to avoid confusion with the variable name.
    return base_command_template.replace("{name_placeholder}", name_of_user)
