import subprocess
import shlex
import logging

logger = logging.getLogger(__name__)

def _validate_inputs(command: str, args: list) -> None:
    if not isinstance(command, str):
        logger.error("Invalid type for command: %r", type(command))
        raise ValueError("Command must be a non-empty string.")
    if not command.strip():
        logger.error("Empty command string provided.")
        raise ValueError("Command must be a non-empty string.")
    if not isinstance(args, list):
        logger.error("Invalid type for args: %r", type(args))
        raise ValueError("Args must be a list.")

def _prepare_command(command: str, args: list) -> list:
    logger.debug("Preparing command with shlex: command=%r, args=%r", command, args)
    try:
        cmd_parts = shlex.split(command)
    except ValueError as e:
        logger.error("shlex.split failed for command=%r: %s", command, e, exc_info=True)
        raise ValueError(f"Invalid command: {e}") from e

    if not args:
        return cmd_parts

    arg_parts = [str(a) for a in args]
    return cmd_parts + arg_parts

def _run_command(full_cmd: list) -> str:
    if not isinstance(full_cmd, list) or not full_cmd:
        logger.error("Full command must be a non-empty list. Got: %r", full_cmd)
        raise ValueError("Full command must be a non-empty list of arguments.")

    safe_cmd_str = " ".join(shlex.quote(str(p)) for p in full_cmd)
    logger.info("Executing command: %s", safe_cmd_str)

    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug("Command executed successfully: %s", safe_cmd_str)
        return result.stdout
    except FileNotFoundError as e:
        logger.error("Command not found: %s", safe_cmd_str, exc_info=True)
        raise ValueError(f"Invalid command: {e}") from e
    except PermissionError as e:
        logger.error("Permission error executing command: %s", safe_cmd_str, exc_info=True)
        raise ValueError(f"Cannot execute command: {e}") from e
    except subprocess.CalledProcessError as e:
        err_output = e.stderr if e.stderr is not None else e.stdout
        logger.error(
            "Command failed with exit code %s: %s\nOutput: %s",
            e.returncode,
            safe_cmd_str,
            err_output,
            exc_info=True
        )
        raise ValueError(f"Command failed with exit code {e.returncode}: {err_output}") from e
    except OSError as e:
        logger.error("OS error during command execution: %s", safe_cmd_str, exc_info=True)
        raise ValueError(f"Execution error: {e}") from e

def execute_shell_command(command: str, args: list) -> str:
    """
    Execute a shell command with user-provided arguments.

    :param command: str - the base command to execute (e.g., "ls" or "git status")
    :param args: list - a list of arguments for the command
    :return: str - the standard output of the executed command
    :raises ValueError: when the command is invalid or cannot be executed
    """
    _validate_inputs(command, args)
    full_cmd = _prepare_command(command, args)
    return _run_command(full_cmd)
