import subprocess
import shlex
import logging

logger = logging.getLogger(__name__)


def _validate_inputs(cmd: str, arguments: list) -> None:
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("Invalid command: 'cmd' must be a non-empty string.")
    if not isinstance(arguments, list):
        raise ValueError("Invalid arguments: 'arguments' must be a list.")


def _split_base_command(cmd: str) -> list:
    try:
        return shlex.split(cmd)
    except ValueError as e:
        raise ValueError(f"Invalid command: {e}") from e


def _assemble_command(cmd: str, arguments: list) -> list:
    base_parts = _split_base_command(cmd)
    arg_parts = [str(a) for a in arguments]
    return base_parts + arg_parts


def _format_command_for_logging(full_cmd: list) -> str:
    # Use shlex.quote to produce a shell-safe representation for logging.
    return " ".join(shlex.quote(part) for part in full_cmd)


def _execute_command(full_cmd: list) -> str:
    cmd_str = _format_command_for_logging(full_cmd)
    logger.info("Executing command: %s", cmd_str)
    try:
        result = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        stdout_text = result.stdout or ""
        stderr_text = result.stderr or ""
        logger.info("Command succeeded: %s", cmd_str)
        if stdout_text.strip():
            logger.info("Command stdout:\n%s", stdout_text.strip())
        else:
            logger.info("Command produced no stdout")
        if stderr_text.strip():
            logger.debug("Command stderr (non-fatal):\n%s", stderr_text.strip())
        return stdout_text
    except FileNotFoundError as e:
        logger.error("Command not found: %s", full_cmd[0])
        raise ValueError(f"Command not found: {full_cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "").strip()
        err = (e.stderr or "").strip()
        logger.error("Command failed (exit code %s): %s", e.returncode, cmd_str)
        if out:
            logger.error("Command stdout:\n%s", out)
        if err:
            logger.error("Command stderr:\n%s", err)
        message = err if err else (out if out else f"Command failed with exit code {e.returncode}")
        raise ValueError(message) from e
    except Exception as e:
        logger.exception("Unexpected error while executing command: %s", cmd_str)
        raise ValueError(str(e)) from e


def run_shell_command(cmd: str, arguments: list) -> str:
    """
    Run a shell command with user-specified arguments.

    Args:
        cmd: The primary command to run (can include subcommands or flags).
        arguments: A list of arguments for the command.

    Returns:
        The standard output from the executed command as a string.

    Raises:
        ValueError: If the command is invalid or fails to execute.
    """
    _validate_inputs(cmd, arguments)
    full_cmd = _assemble_command(cmd, arguments)
    return _execute_command(full_cmd)
