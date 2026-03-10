def load_local_file(file_name, allowed_files):
    if file_name not in allowed_files:
        return None
    with open(file_name, 'r') as file:
        return file.read()
