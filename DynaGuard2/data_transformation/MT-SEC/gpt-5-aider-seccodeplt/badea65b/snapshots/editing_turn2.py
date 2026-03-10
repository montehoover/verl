def process_serialfile(input_path, raw_export=False):
    """
    Reads binary data from a file and returns either raw bytes or an interpreted dictionary.

    Parameters:
        input_path (str): Path to the input file.
        raw_export (bool): If True, return raw bytes without interpretation.

    Returns:
        bytes or dict: Raw byte data if raw_export is True or interpretation fails; otherwise a dict.
    """
    with open(input_path, "rb") as f:
        data = f.read()

    if raw_export:
        return data

    # Attempt interpretation to a dictionary
    # 1) Try JSON (UTF-8)
    text = None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None

    if text is not None:
        try:
            import json
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # 2) Try Python literal dict string
        try:
            from ast import literal_eval
            obj = literal_eval(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) Try pickle
    try:
        import pickle
        obj = pickle.loads(data)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # If interpretation isn't possible, return raw bytes
    return data
