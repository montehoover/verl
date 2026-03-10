def extract_fields(data: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specified fields from a list of dictionaries.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.

    Returns:
        A new list of dictionaries, each containing only the specified fields.

    Raises:
        ValueError: If a specified field is not found in a dictionary.
    """
    result = []
    for record in data:
        new_record = {}
        for field in fields:
            if field not in record:
                raise ValueError(f"Field '{field}' not found in record: {record}")
            new_record[field] = record[field]
        result.append(new_record)
    return result
