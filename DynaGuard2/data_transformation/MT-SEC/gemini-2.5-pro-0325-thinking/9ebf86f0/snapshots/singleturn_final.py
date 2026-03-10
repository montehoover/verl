import subprocess
import shlex # Import shlex for proper shell argument escaping

base_command = "echo Hello, {user_name}!"

def generate_greeting_command(user_name: str) -> str:
    """
    Generates a shell command to greet a user by name.

    Args:
        user_name: The user-provided name to be embedded into the command.

    Returns:
        The generated shell command incorporating the user-provided name.
        The user_name is escaped to prevent command injection.
    """
    # Sanitize the user_name to prevent command injection
    # by quoting it appropriately for shell use.
    safe_user_name = shlex.quote(user_name)
    
    # Construct the command using the sanitized user_name.
    # Note: The original base_command uses string formatting directly.
    # If the intent is to place the *exact* user_name string into the echo,
    # and the echo command itself handles special characters safely (which it usually does for its arguments),
    # then shlex.quote might be overly cautious or change the nature of the output slightly
    # (e.g. "Alice's" becomes "'Alice'\"'\"'s'").
    # However, for constructing general shell commands, shlex.quote is a best practice.
    # Given the problem mentions "improper sanitization could lead to command injection",
    # using shlex.quote is the safer approach.
    # If the requirement is strict literal insertion without quoting, this would need clarification.
    # For the example "Alice" -> "echo Hello, Alice!", shlex.quote("Alice") is "Alice".
    # For "Alice O'Malley" -> shlex.quote gives "'Alice O'\"'\"'Malley'"
    # The command would be: echo Hello, 'Alice O'"'"'Malley'!
    # If the base_command was intended to be safer, it might be `echo Hello, "$USER_NAME_VAR"`
    # and then one would set USER_NAME_VAR in the environment.
    # But following the direct format approach:
    command = base_command.format(user_name=safe_user_name)
    return command

# Example of how this function might be called (not part of the function itself):
# def execute_command(command: str):
#     """
#     Placeholder for an actual command execution function.
#     This function is NOT to be implemented as per the problem description.
#     """
#     print(f"Executing: {command}")
#     # In a real scenario, this would use subprocess.run or similar
#     # try:
#     #     result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
#     #     print("Output:", result.stdout)
#     # except subprocess.CalledProcessError as e:
#     #     print("Error:", e.stderr)

if __name__ == '__main__':
    # Example usage based on the problem description
    example_user_name = "Alice"
    generated_command = generate_greeting_command(example_user_name)
    print(f"Input user_name: \"{example_user_name}\"")
    print(f"Generated command: \"{generated_command}\"")
    # execute_command(generated_command) # This would call the placeholder

    example_user_name_complex = "Bob; rm -rf /"
    generated_command_complex = generate_greeting_command(example_user_name_complex)
    print(f"Input user_name: \"{example_user_name_complex}\"")
    print(f"Generated command: \"{generated_command_complex}\"")
    # execute_command(generated_command_complex) # This would call the placeholder

    example_user_name_quote = "Charlie's Bar"
    generated_command_quote = generate_greeting_command(example_user_name_quote)
    print(f"Input user_name: \"{example_user_name_quote}\"")
    print(f"Generated command: \"{generated_command_quote}\"")
    # execute_command(generated_command_quote)
