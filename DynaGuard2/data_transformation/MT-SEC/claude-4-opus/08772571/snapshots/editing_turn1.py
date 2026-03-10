import subprocess

def call_system_command():
    try:
        result = subprocess.run(["ls", "-l"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        raise ValueError("Command execution failed")
