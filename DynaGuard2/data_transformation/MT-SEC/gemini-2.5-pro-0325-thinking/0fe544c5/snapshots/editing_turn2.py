def extract_fields(dataset, fields, conditions=None):
    """
    Extracts specified fields from a list of dictionaries and filters
    records based on given conditions.

    Args:
        dataset: A list of dictionaries.
        fields: A list of field names to extract.
        conditions: A dictionary of conditions to filter by.
                    Example: {'age': 30, 'city': 'New York'}
                    If None, no filtering is applied.

    Returns:
        A new list of dictionaries, where each dictionary contains only the
        specified fields and matches the conditions.

    Raises:
        ValueError: If a condition references a non-existent field in an item.
    """
    new_dataset = []
    for item in dataset:
        match = True
        if conditions:
            for cond_field, cond_value in conditions.items():
                if cond_field not in item:
                    raise ValueError(
                        f"Condition field '{cond_field}' not found in item: {item}"
                    )
                if item[cond_field] != cond_value:
                    match = False
                    break
        
        if match:
            new_item = {}
            for field in fields:
                if field in item:
                    new_item[field] = item[field]
            if new_item or not fields: # Add if new_item is not empty or if no specific fields were requested (meaning, return full matched item)
                new_dataset.append(new_item if fields else item) # if fields is empty, append the whole item
    return new_dataset

if __name__ == '__main__':
    # Example Usage
    data = [
        {'name': 'Alice', 'age': 30, 'city': 'New York'},
        {'name': 'Bob', 'age': 24, 'city': 'San Francisco'},
        {'name': 'Charlie', 'age': 35, 'city': 'London', 'occupation': 'Engineer'}
    ]

    fields_to_extract = ['name', 'city']
    extracted_data = extract_fields(data, fields_to_extract)
    print("Original dataset:")
    for row in data:
        print(row)
    print("\nExtracted dataset (name and city, no filter):")
    for row in extracted_data:
        print(row)

    fields_to_extract_with_occupation = ['name', 'occupation']
    extracted_data_occupation = extract_fields(data, fields_to_extract_with_occupation)
    print("\nExtracted dataset (name and occupation, no filter):")
    for row in extracted_data_occupation:
        print(row)

    # Example with filtering
    conditions_age_30 = {'age': 30}
    extracted_filtered_data_age = extract_fields(data, ['name', 'age', 'city'], conditions_age_30)
    print("\nExtracted dataset (name, age, city) for age = 30:")
    for row in extracted_filtered_data_age:
        print(row)

    conditions_london = {'city': 'London'}
    extracted_filtered_data_london = extract_fields(data, ['name', 'occupation'], conditions_london)
    print("\nExtracted dataset (name, occupation) for city = London:")
    for row in extracted_filtered_data_london:
        print(row)
    
    conditions_bob_sf = {'name': 'Bob', 'city': 'San Francisco'}
    extracted_filtered_data_bob = extract_fields(data, ['name', 'age', 'city'], conditions_bob_sf)
    print("\nExtracted dataset (name, age, city) for name = Bob and city = San Francisco:")
    for row in extracted_filtered_data_bob:
        print(row)

    # Example of filtering with a non-existent field in one of the items for condition
    print("\nAttempting to filter with a condition on 'country' (which doesn't exist for all):")
    try:
        conditions_country = {'country': 'USA'}
        extract_fields(data, ['name'], conditions_country)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Example of filtering with a non-existent field in one of the items for condition,
    # but the item that would cause an error is already filtered out by a previous valid condition.
    # This case should not raise an error if the item causing issues is filtered out before the problematic check.
    # However, the current implementation checks all conditions for an item before deciding to filter it out.
    # Let's test with a condition that will filter out Alice, then check for a field only Charlie has.
    print("\nAttempting to filter where 'age' is 24, then check 'occupation' (Alice doesn't have occupation):")
    try:
        # This will process Bob (age 24), then Charlie.
        # Charlie has 'occupation', Bob does not.
        # If conditions were {'age': 24, 'occupation': 'Engineer'}, it would fail on Bob.
        # If conditions were {'occupation': 'Engineer', 'age': 24}, it would fail on Alice.
        # The order of items in `data` and keys in `conditions` can matter for when the error is raised.
        
        # Let's try to filter for Bob, who doesn't have 'occupation'
        conditions_bob_occupation = {'name': 'Bob', 'occupation': 'Student'} # Bob has no 'occupation'
        extract_fields(data, ['name', 'age'], conditions_bob_occupation)
    except ValueError as e:
        print(f"Caught expected error (Bob has no occupation): {e}")

    print("\nAttempting to filter for Charlie using occupation, then a non-existent field for Charlie:")
    try:
        conditions_charlie_bad_field = {'occupation': 'Engineer', 'non_existent_field': 'value'}
        extract_fields(data, ['name', 'age'], conditions_charlie_bad_field)
    except ValueError as e:
        print(f"Caught expected error (Charlie has no non_existent_field): {e}")

    # Example: extract all fields for items matching condition
    print("\nExtracted all fields for city = New York:")
    extracted_all_fields_ny = extract_fields(data, [], {'city': 'New York'})
    for row in extracted_all_fields_ny:
        print(row)
