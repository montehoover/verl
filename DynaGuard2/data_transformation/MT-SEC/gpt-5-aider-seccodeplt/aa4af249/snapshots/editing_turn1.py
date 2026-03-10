def check_fields(dataset, field_names):
    """
    Return True if any of the provided field_names appear as keys
    in any record (dict-like) in the dataset; otherwise False.

    Parameters:
    - dataset: iterable of dictionaries (or dict-like objects)
    - field_names: iterable of field names to look for

    Notes:
    - If field_names is empty or None, returns False.
    - Non-mapping items in the dataset are ignored.
    """
    if not field_names:
        return False

    target_fields = set(field_names)
    if not target_fields:
        return False

    all_fields = set()
    if dataset:
        for record in dataset:
            if hasattr(record, "keys"):
                all_fields.update(record.keys())

    return not all_fields.isdisjoint(target_fields)
