def extract_fields(data: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specified fields from a list of dictionaries.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.

    Returns:
        A new list of dictionaries, where each dictionary contains only the
        specified fields from the original dictionaries.
    """
    result = []
    for record in data:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        result.append(new_record)
    return result
