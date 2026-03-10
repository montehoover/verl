def execute_shell_command(directory: str, command: str = "ls") -> str:
    import subprocess
    import shlex

    if not isinstance(command, str) or not command.strip():
        raise ValueError("A valid command must be provided.")

    cmd = ""
    is_ls = command.strip() == "ls"

    if is_ls:
        if not isinstance(directory, str) or not directory:
            raise ValueError("A valid directory path must be provided.")
        cmd = f"ls {shlex.quote(directory)}"
    else:
        cmd = command

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        raise ValueError(str(exc))

    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "command failed")

    if is_ls:
        filenames = [line for line in result.stdout.splitlines() if line]
        return ",".join(filenames)

    return result.stdout
