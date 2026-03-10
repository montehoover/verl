def merge_dicts(original: dict, updates: dict) -> dict:
    """
    Merges two dictionaries, applying updates to the original.

    Args:
        original: The base dictionary.
        updates: A dictionary containing updates to be applied.

    Returns:
        A new dictionary with the updates applied.
    """
    merged = original.copy()  # Start with a copy of the original
    merged.update(updates)  # Apply updates, overwriting existing keys if necessary
    return merged


def restricted_update(original: dict, updates: dict, allowed_fields: list[str]) -> dict:
    """
    Merges two dictionaries, applying updates to the original only for allowed fields.

    Args:
        original: The base dictionary.
        updates: A dictionary containing updates to be applied.
        allowed_fields: A list of keys that are allowed to be updated.

    Returns:
        A new dictionary with the restricted updates applied.
    """
    merged = original.copy()  # Start with a copy of the original
    for key, value in updates.items():
        if key in allowed_fields:
            merged[key] = value
    return merged

if __name__ == '__main__':
    # Example Usage
    dict1 = {'a': 1, 'b': 2, 'c': 3}
    dict2 = {'b': 20, 'd': 40}
    
    merged_dict = merge_dicts(dict1, dict2)
    print(f"Original dictionary 1: {dict1}")
    print(f"Updates dictionary 2: {dict2}")
    print(f"Merged dictionary: {merged_dict}")

    dict3 = {'name': 'Alice', 'age': 30}
    dict4 = {'age': 31, 'city': 'New York'}
    merged_dict_2 = merge_dicts(dict3, dict4)
    print(f"Original dictionary 3: {dict3}")
    print(f"Updates dictionary 4: {dict4}")
    print(f"Merged dictionary 2: {merged_dict_2}")

    # Example Usage for restricted_update
    dict5 = {'name': 'Bob', 'age': 25, 'city': 'London', 'occupation': 'Engineer'}
    dict6 = {'age': 26, 'city': 'Paris', 'occupation': 'Artist'}
    allowed = ['age', 'city']
    
    restricted_merged_dict = restricted_update(dict5, dict6, allowed)
    print(f"\nOriginal dictionary 5: {dict5}")
    print(f"Updates dictionary 6: {dict6}")
    print(f"Allowed fields for update: {allowed}")
    print(f"Restricted merged dictionary: {restricted_merged_dict}")

    allowed_only_occupation = ['occupation']
    restricted_merged_dict_2 = restricted_update(dict5, dict6, allowed_only_occupation)
    print(f"\nOriginal dictionary 5: {dict5}")
    print(f"Updates dictionary 6: {dict6}")
    print(f"Allowed fields for update: {allowed_only_occupation}")
    print(f"Restricted merged dictionary 2: {restricted_merged_dict_2}")
