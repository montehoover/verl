from collections import defaultdict

def select_fields(records: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specific fields from a list of records.

    Args:
        records: A list of dictionaries.
        fields: A list of field names (strings) to extract.

    Returns:
        A list of dictionaries, where each dictionary contains only the
        specified fields from the original record. If a field is not
        present in a record, it is omitted from the result for that record.
    """
    selected_records = []
    for record in records:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        selected_records.append(new_record)
    return selected_records
