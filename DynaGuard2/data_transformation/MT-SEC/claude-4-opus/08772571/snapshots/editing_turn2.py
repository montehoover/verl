import subprocess

def call_system_command(options=None):
    try:
        command = ["ls"]
        if options:
            command.extend(options)
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        raise ValueError("Command execution failed")
