def check_fields(dataset: list[dict], field_names: list[str]) -> bool:
    """
    Checks if any of the provided field names exist as keys in any of the dictionaries within the dataset.

    Args:
        dataset: A list of dictionaries.
        field_names: A list of field names (strings) to check for.

    Returns:
        True if at least one field name is found in at least one dictionary, False otherwise.
    """
    for record in dataset:
        for field_name in field_names:
            if field_name in record:
                return True
    return False
