def construct_command(base_command: str, parameters: list[str]) -> str:
    """
    Constructs a command string from a base command and a list of parameters.

    Args:
        base_command: The base command string.
        parameters: A list of parameter strings.

    Returns:
        A single string representing the full command.
    """
    return base_command + " " + " ".join(parameters)

if __name__ == '__main__':
    # Example usage:
    cmd = construct_command("ls", ["-l", "-a", "/tmp"])
    print(f"Constructed command: '{cmd}'")

    cmd2 = construct_command("docker run", ["-it", "--rm", "ubuntu:latest", "/bin/bash"])
    print(f"Constructed command: '{cmd2}'")

    cmd3 = construct_command("git", ["commit", "-m", "Initial commit"])
    print(f"Constructed command: '{cmd3}'")
