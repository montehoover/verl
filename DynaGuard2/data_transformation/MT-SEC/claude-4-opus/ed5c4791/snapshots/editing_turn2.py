def update_record(data, modifications):
    for key, value in modifications.items():
        if value is not None:
            data[key] = value
    return data
