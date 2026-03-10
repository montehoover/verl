def execute_shell_command(directory: str) -> None:
    import subprocess
    import shlex

    if not isinstance(directory, str) or not directory:
        raise ValueError("A valid directory path must be provided.")

    cmd = f"ls {shlex.quote(directory)}"
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
        raise ValueError(result.stderr.strip() or "ls command failed")

    filenames = [line for line in result.stdout.splitlines() if line]
    print(",".join(filenames))
