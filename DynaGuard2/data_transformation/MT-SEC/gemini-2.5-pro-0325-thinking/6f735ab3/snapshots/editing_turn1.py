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
