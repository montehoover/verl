def generate_announcement(title: str, message: str) -> str:
    """
    Create a shell command that echoes an announcement combining title and message.
    Returns a POSIX-safe echo command string.
    """
    def _sh_single_quote(s: str) -> str:
        # Safely single-quote a string for POSIX shells
        return "'" + s.replace("'", "'\"'\"'") + "'"

    content = f"{title}: {message}"
    return f"echo {_sh_single_quote(content)}"
