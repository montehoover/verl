import shlex
import subprocess

base_command = "echo Hello, {user_name}!"

def construct_command(command_template: str) -> str:
    """
    Constructs a shell command from a template.

    Args:
        command_template: The template string for the command.

    Returns:
        The shell command string.
    """
    return command_template

def embed_user_name(command_template: str, user_name: str) -> str:
    """
    Embeds a sanitized user name into a command template.

    Args:
        command_template: The template string for the command,
                          e.g., "echo 'Hello, {name}!'"
        user_name: The user name to embed.

    Returns:
        The shell command string with the user name embedded.
    """
    sanitized_user_name = shlex.quote(user_name)
    return command_template.format(name=sanitized_user_name)

def generate_greeting_command(user_name: str) -> str:
    """
    Generates a shell command to greet a user.

    Args:
        user_name: The name of the user to greet.

    Returns:
        A shell command string.
    """
    sanitized_user_name = shlex.quote(user_name)
    return base_command.format(user_name=sanitized_user_name)

if __name__ == '__main__':
    # Example usage for construct_command:
    template = "ls -l /tmp"
    command = construct_command(template)
    print(f"Generated command: {command}")

    template_with_placeholder = "echo 'Hello, {name}!'"
    # This basic version doesn't do placeholder replacement yet.
    command_with_placeholder = construct_command(template_with_placeholder)
    print(f"Generated command with placeholder (construct_command): {command_with_placeholder}")

    # Example usage for embed_user_name:
    user_template = "echo 'Greetings, {name}!'"
    user_name_input = "Alice"
    command_with_user = embed_user_name(user_template, user_name_input)
    print(f"Generated command with user name: {command_with_user}")

    user_name_malicious = "Alice; rm -rf /"
    command_with_malicious_user = embed_user_name(user_template, user_name_malicious)
    print(f"Generated command with sanitized malicious user name (embed_user_name): {command_with_malicious_user}")

    # Example usage for generate_greeting_command:
    greeting_user_name = "Bob"
    greeting_command = generate_greeting_command(greeting_user_name)
    print(f"Generated greeting command: {greeting_command}")
    # Example of running the generated command (optional, for demonstration)
    # try:
    #     subprocess.run(greeting_command, shell=True, check=True, text=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error executing command: {e}")

    greeting_user_name_malicious = "Charlie; date"
    greeting_command_malicious = generate_greeting_command(greeting_user_name_malicious)
    print(f"Generated greeting command with sanitized malicious user name: {greeting_command_malicious}")
