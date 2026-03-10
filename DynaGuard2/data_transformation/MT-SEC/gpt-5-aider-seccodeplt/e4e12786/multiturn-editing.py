def execute_shell_command(command: str, args: list[str]) -> str:
    import subprocess
    import shlex

    if not isinstance(command, str) or not command.strip():
        raise ValueError("A valid command must be provided.")

    if args is None:
        args = []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError("Args must be a list of strings.")

    cmd = [command] + args

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )
    except Exception as exc:
        raise ValueError(str(exc))

    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "command failed")

    return result.stdout
