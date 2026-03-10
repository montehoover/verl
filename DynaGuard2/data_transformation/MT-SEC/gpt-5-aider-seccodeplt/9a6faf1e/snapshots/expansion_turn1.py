from typing import Union
import os


def read_file_contents(path: Union[str, os.PathLike]) -> str:
    """
    Read the file at the given path and return its raw contents as a string.

    The file is read as bytes and decoded using UTF-8 with 'surrogateescape'
    to preserve any undecodable bytes, ensuring a lossless round-trip if needed.
    """
    file_path = os.fspath(path)
    with open(file_path, 'rb') as f:
        data = f.read()
    return data.decode('utf-8', errors='surrogateescape')
