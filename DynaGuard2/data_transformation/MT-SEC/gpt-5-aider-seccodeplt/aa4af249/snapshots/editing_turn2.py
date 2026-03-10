def check_fields(dataset, field_names, conditions=None):
    """
    Return a boolean based on field presence, optionally filtered by conditions.

    - If 'conditions' is provided (dict), only records where all condition
      key/value pairs match are considered.
    - If 'field_names' is provided and non-empty, returns True if any of the
      considered records contain at least one of the specified field names.
    - If 'field_names' is empty or None but 'conditions' are provided, returns
      True if any record matches the conditions.
    - If both 'field_names' and 'conditions' are empty/None, returns False.

    Parameters:
    - dataset: iterable of dictionaries (or dict-like objects)
    - field_names: iterable of field names to look for
    - conditions: dict of field -> value that records must match
    """
    if dataset is None:
        return False

    target_fields = set(field_names) if field_names else set()
    cond_items = tuple(conditions.items()) if conditions else ()

    # If no constraints at all, nothing to check
    if not target_fields and not cond_items:
        return False

    for record in dataset:
        if not hasattr(record, "keys"):
            continue

        # Apply conditions if provided
        if cond_items and not all((k in record and record[k] == v) for k, v in cond_items):
            continue

        # If we're only filtering by conditions and no field constraint
        if not target_fields:
            return True

        # Check for presence of any target field in the (filtered) record
        if not set(record.keys()).isdisjoint(target_fields):
            return True

    return False
