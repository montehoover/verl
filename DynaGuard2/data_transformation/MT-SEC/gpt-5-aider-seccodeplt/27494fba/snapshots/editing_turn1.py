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
