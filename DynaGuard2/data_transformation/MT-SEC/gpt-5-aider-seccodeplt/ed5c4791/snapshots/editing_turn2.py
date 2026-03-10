def update_record(data, modifications):
    merged = dict(data)
    for key, value in modifications.items():
        if value is not None:
            merged[key] = value
    return merged
