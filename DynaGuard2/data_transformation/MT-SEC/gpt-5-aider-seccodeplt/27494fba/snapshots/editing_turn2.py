def update_record(record, updates):
    """
    Update the given record dictionary with values from updates and return it.

    Args:
        record (dict): The original dictionary to update.
        updates (dict): A dictionary containing keys and values to merge into record.

    Returns:
        dict: The updated dictionary with values from updates applied.
    """
    record.update(updates)
    return record


def record_summary(record):
    """
    Return a string summary with the number of keys and a comma-separated list
    of field names (dictionary keys) sorted alphabetically.

    Args:
        record (dict): Dictionary to summarize.

    Returns:
        str: Summary string, e.g., "3 keys: a, b, c".
    """
    keys = [str(k) for k in record.keys()]
    keys_sorted = sorted(keys, key=lambda s: s.lower())
    return f"{len(record)} keys: {', '.join(keys_sorted)}"
