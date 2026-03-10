def filter_and_extract(data: list[dict], fields: list[str], filter_conditions: dict) -> list[dict]:
    """
    Filters records based on specified conditions and then extracts specified
    fields from the matching records.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.
        filter_conditions: A dictionary specifying the field and the value
                           it must match. For example, {'status': 'active'}.

    Returns:
        A new list of dictionaries, where each dictionary contains only the
        specified fields from the records that matched the filter conditions.
    """
    result = []
    for record in data:
        match = True
        for key, value in filter_conditions.items():
            if record.get(key) != value:
                match = False
                break
        
        if match:
            new_record = {}
            for field in fields:
                if field in record:
                    new_record[field] = record[field]
            if new_record:  # Only add if there are fields to extract or if fields list is empty
                result.append(new_record)
    return result
