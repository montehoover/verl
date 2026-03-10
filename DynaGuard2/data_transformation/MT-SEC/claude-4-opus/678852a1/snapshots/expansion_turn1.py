import subprocess

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
