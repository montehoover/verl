def check_fields(dataset: list[dict], field_names: list[str], conditions: dict) -> bool:
    """
    Checks if any record in the dataset meets criteria based on field presence and conditions.
    A record is considered a match if:
    1. It contains at least one of the field names specified in `field_names`.
       (If `field_names` is empty, this criterion cannot be met by any record).
    2. AND it satisfies all key-value pair conditions specified in `conditions`.
       (If `conditions` is empty, this criterion is considered met for any record that passes criterion 1).

    Args:
        dataset: A list of dictionaries.
        field_names: A list of field names (strings). A record must contain at least one of these.
                     If empty, no record can meet the first criterion, and the function will return False.
        conditions: A dictionary specifying field-value pairs to match.
                    Example: {"age": 30, "city": "New York"}.
                    If empty, this part of the check is trivially satisfied.

    Returns:
        True if at least one record meets both criteria (field presence AND conditions), False otherwise.
    """
    for record in dataset:
        # Criterion 1: Record must contain at least one of the field_names
        has_matching_field = False
        # If field_names is empty, this loop won't run, has_matching_field remains False.
        for fn in field_names:
            if fn in record:
                has_matching_field = True
                break
        
        if not has_matching_field:
            continue  # This record doesn't meet the first criterion.

        # Criterion 2: Record must satisfy all conditions
        # If conditions is empty, this part is considered met.
        satisfies_all_conditions = True
        if conditions:  # Only iterate if conditions is not empty
            for cond_key, cond_value in conditions.items():
                if record.get(cond_key) != cond_value:
                    satisfies_all_conditions = False
                    break
        
        if satisfies_all_conditions:  # Implicitly, has_matching_field was also true to reach here
            return True  # Found a record meeting both criteria

    return False  # No record found that meets both criteria
