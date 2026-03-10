from collections import defaultdict

__all__ = ["select_fields"]


def select_fields(records, fields):
    """
    Extract specified fields from a list of record dictionaries.

    - records: list of dict objects (each record).
    - fields: list of strings specifying field names to extract.

    Returns a new list of dicts containing only the requested fields.
    Missing fields in a record are included with value None.
    """
    if records is None:
        return []

    if not isinstance(fields, (list, tuple)):
        raise TypeError("fields must be a list or tuple of field names")

    result = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each record must be a dict")
        rec_with_default = defaultdict(lambda: None, record)
        result.append({field: rec_with_default[field] for field in fields})
    return result
