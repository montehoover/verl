def update_record(data, modifications):
    merged = dict(data)
    merged.update(modifications)
    return merged
