def is_allowed_command(cmd: str) -> bool:
    """
    Return True if `cmd` is in the list of allowed shell commands.

    The allowed commands are defined locally in this function.
    """
    if not isinstance(cmd, str):
        return False

    allowed_commands = {
        "ls",
        "pwd",
        "echo",
        "cat",
        "grep",
        "cd",
        "touch",
        "mkdir",
        "rm",
        "rmdir",
        "cp",
        "mv",
        "chmod",
        "chown",
        "head",
        "tail",
        "less",
        "more",
        "find",
        "which",
        "whoami",
        "date",
        "uname",
        "env",
        "printenv",
        "ps",
        "kill",
        "df",
        "du",
        "tar",
        "gzip",
        "gunzip",
        "ssh",
        "scp",
        "curl",
        "wget",
    }

    normalized = cmd.strip()
    return normalized in allowed_commands
