def read_file_content(file_path):
    with open(file_path, 'rb') as file:
        return file.read()
