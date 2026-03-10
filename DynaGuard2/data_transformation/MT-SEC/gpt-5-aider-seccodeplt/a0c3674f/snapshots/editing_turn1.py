from typing import Any, Dict

def update_record(data: Dict[Any, Any], modifications: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Return a new dictionary combining data with modifications.

    - Shallow merge: keys in modifications overwrite keys in data.
    - Does not mutate the original 'data' dictionary.
    """
    updated = dict(data)
    updated.update(modifications)
    return updated
