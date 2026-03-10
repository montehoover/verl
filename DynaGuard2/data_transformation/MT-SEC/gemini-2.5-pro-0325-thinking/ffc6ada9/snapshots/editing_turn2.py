import re

def construct_command(base_command: str, parameters: list[str]) -> str:
    """
    Constructs a command string from a base command and a list of parameters.
    Validates the command and parameters for potentially unsafe characters.

    Args:
        base_command: The base command string.
        parameters: A list of parameter strings.

    Returns:
        A single string representing the full command.

    Raises:
        ValueError: If the base_command is empty, or if the base_command or
                    any parameter contains invalid characters (e.g., null bytes).
    """
    if not base_command:
        raise ValueError("Base command cannot be empty.")

    # Check for null bytes in base_command
    if '\0' in base_command:
        raise ValueError("Base command contains invalid null bytes.")

    # Validate base_command against a stricter set of allowed characters
    # This regex allows alphanumeric, underscores, hyphens, periods, and forward slashes.
    # It's a basic example; real-world command validation can be much more complex.
    if not re.match(r"^[a-zA-Z0-9_./-]+$", base_command.split()[0]): # Check only the command part, not its own args
        raise ValueError(f"Base command '{base_command.split()[0]}' contains invalid characters.")

    validated_parameters = []
    for param in parameters:
        if '\0' in param:
            raise ValueError("Parameters contain invalid null bytes.")
        # Add more specific parameter validation if needed.
        # For now, we'll allow most characters in parameters, as they might be paths, quoted strings, etc.
        # However, be cautious with shell metacharacters if not properly handled/escaped.
        validated_parameters.append(param)

    # Join the base command and validated parameters
    full_command_parts = [base_command] + validated_parameters
    return " ".join(full_command_parts)

if __name__ == '__main__':
    # Example usage:
    try:
        cmd = construct_command("ls", ["-l", "-a", "/tmp"])
        print(f"Constructed command: '{cmd}'")

        cmd2 = construct_command("docker run", ["-it", "--rm", "ubuntu:latest", "/bin/bash"])
        print(f"Constructed command: '{cmd2}'")

        cmd3 = construct_command("git", ["commit", "-m", "Initial commit"])
        print(f"Constructed command: '{cmd3}'")

        # Example of a command that should raise ValueError (empty base command)
        try:
            construct_command("", ["-l"])
        except ValueError as e:
            print(f"Error (expected): {e}")

        # Example of a command that should raise ValueError (invalid char in base command)
        try:
            construct_command("ls;rm -rf /", ["-l"])
        except ValueError as e:
            print(f"Error (expected): {e}")

        # Example of a command that should raise ValueError (null byte in parameter)
        try:
            construct_command("echo", ["hello\0world"])
        except ValueError as e:
            print(f"Error (expected): {e}")

        # Example of a valid command with path
        cmd4 = construct_command("/usr/bin/my-script.sh", ["--input", "./data/file.txt"])
        print(f"Constructed command: '{cmd4}'")

    except ValueError as e:
        print(f"An unexpected error occurred: {e}")
