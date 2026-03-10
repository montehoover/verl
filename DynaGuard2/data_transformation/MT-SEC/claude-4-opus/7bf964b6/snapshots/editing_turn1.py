import os
import subprocess

def validate_and_execute_command(shell_cmd):
    try:
        result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command execution failed with return code {e.returncode}: {e.stderr}")
    except Exception as e:
        raise ValueError(f"Command execution failed: {str(e)}")
