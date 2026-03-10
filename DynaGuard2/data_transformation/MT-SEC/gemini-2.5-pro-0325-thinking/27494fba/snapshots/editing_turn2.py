def update_record(record: dict, updates: dict) -> dict:
    """
    Updates a record dictionary with new values.

    Args:
        record: The original dictionary.
        updates: A dictionary containing keys and values to update in the record.

    Returns:
        A new dictionary with the updates merged into the record.
    """
    updated_record = record.copy()
    updated_record.update(updates)
    return updated_record

def record_summary(record: dict) -> str:
    """
    Generates a summary string for a record dictionary.

    Args:
        record: The dictionary to summarize.

    Returns:
        A string summary with the number of keys and a comma-separated list
        of field names, sorted alphabetically.
    """
    num_keys = len(record)
    sorted_keys = sorted(record.keys())
    field_names = ", ".join(sorted_keys)
    return f"Record has {num_keys} keys: {field_names}"
