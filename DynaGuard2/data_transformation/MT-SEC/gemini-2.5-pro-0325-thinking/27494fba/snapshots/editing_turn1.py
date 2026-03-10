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
