import subprocess

ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def execute_command(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: Command failed with exit code {e.returncode}\n{e.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"

def is_command_allowed(command):
    return command in ALLOWED_COMMANDS
