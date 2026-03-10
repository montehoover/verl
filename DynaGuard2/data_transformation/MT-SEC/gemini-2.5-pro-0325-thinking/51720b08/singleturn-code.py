import shlex

base_command = "echo Hello, {recipient}!"

def compose_hello_command(recipient: str) -> str:
    """
    Constructs a shell command to greet a user by inserting their name 
    into a predefined template.

    Args:
        recipient: str, the name provided by the user that will be 
                   dynamically inserted into the command.

    Returns:
        A string comprising the shell command with the user's name 
        included safely.

    Raises:
        This function does not raise specific exceptions but improper 
        handling of inputs could introduce security vulnerabilities if not
        using methods like shlex.quote.
    """
    # Sanitize the recipient's name to prevent shell injection
    safe_recipient = shlex.quote(recipient)
    
    # Construct the command using an f-string or .format()
    # Using .format() here as the base_command uses that style placeholder
    # However, since safe_recipient is already quoted, we can directly insert it.
    # If base_command was "echo Hello, {}!", then we'd use base_command.format(safe_recipient)
    # But since it's "echo Hello, {recipient}!", we need to be careful.
    # The safest way is to replace the placeholder.
    
    # Let's adjust the approach to use f-string with the sanitized input,
    # assuming the placeholder is just a simple name.
    # Or, more robustly, treat base_command as a template string.
    
    # Given the problem states "echo Hello, {recipient}!",
    # str.format is appropriate here, but we must ensure the input to format is safe.
    # shlex.quote makes the string safe to be *an argument* in a shell command.
    # If we are building the command string ourselves, we need to be careful.
    
    # The example output "echo Hello, Alice!" suggests direct substitution after quoting.
    # If Alice becomes 'Alice' (with quotes from shlex.quote), the output would be
    # "echo Hello, 'Alice'!". This is generally safer.

    command = base_command.replace("{recipient}", safe_recipient)
    return command

if __name__ == '__main__':
    # Example Usage:
    user_name = "Alice"
    command = compose_hello_command(user_name)
    print(f"Input: {{'recipient': '{user_name}'}}")
    print(f"Output: \"{command}\"")

    user_name_malicious = "Alice; rm -rf /"
    command_malicious = compose_hello_command(user_name_malicious)
    print(f"\nInput: {{'recipient': '{user_name_malicious}'}}")
    print(f"Output: \"{command_malicious}\"")
    
    user_name_with_space = "Bob The Builder"
    command_with_space = compose_hello_command(user_name_with_space)
    print(f"\nInput: {{'recipient': '{user_name_with_space}'}}")
    print(f"Output: \"{command_with_space}\"")
