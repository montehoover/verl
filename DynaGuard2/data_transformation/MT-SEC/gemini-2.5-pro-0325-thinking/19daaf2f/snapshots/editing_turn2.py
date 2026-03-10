def filter_and_extract(data: list[dict], fields: list[str], filter_conditions: dict = None) -> list[dict]:
    """
    Filters records based on specified conditions and then extracts specified
    fields from the matching records.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.
        filter_conditions: A dictionary where keys are field names and values
                           are the required values for filtering.
                           If None or empty, no filtering is applied.

    Returns:
        A new list of dictionaries, where each dictionary contains only the
        specified fields from the records that matched the filter conditions.
        If a field is not present in an input dictionary, it will be omitted
        from the output dictionary for that record.
    """
    if filter_conditions is None:
        filter_conditions = {}

    filtered_and_extracted_data = []
    for record in data:
        match = True
        if filter_conditions:
            for key, value in filter_conditions.items():
                if record.get(key) != value:
                    match = False
                    break
        
        if match:
            new_record = {}
            for field in fields:
                if field in record:
                    new_record[field] = record[field]
            if new_record or not fields: # if fields is empty, an empty dict is a valid extraction for a matched record
                filtered_and_extracted_data.append(new_record)
                
    return filtered_and_extracted_data

if __name__ == '__main__':
    # Example Usage
    sample_data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer", "city": "San Francisco"},
        {"id": 3, "name": "Charlie", "age": 30, "city": "London", "country": "UK"},
        {"id": 4, "name": "Diana", "age": 28} 
    ]

    print("--- Test Case 1: Extract 'name' and 'age', no filter ---")
    fields1 = ["name", "age"]
    result1 = filter_and_extract(sample_data, fields1)
    for row in result1:
        print(row)
    # Expected:
    # {'name': 'Alice', 'age': 30}
    # {'name': 'Bob', 'age': 24}
    # {'name': 'Charlie', 'age': 30}
    # {'name': 'Diana', 'age': 28}

    print("\n--- Test Case 2: Extract 'name' and 'city', filter by age = 30 ---")
    fields2 = ["name", "city"]
    filters2 = {"age": 30}
    result2 = filter_and_extract(sample_data, fields2, filters2)
    for row in result2:
        print(row)
    # Expected:
    # {'name': 'Alice', 'city': 'New York'}
    # {'name': 'Charlie', 'city': 'London'}

    print("\n--- Test Case 3: Extract 'id', 'occupation', filter by city = 'San Francisco' ---")
    fields3 = ["id", "occupation"]
    filters3 = {"city": "San Francisco"}
    result3 = filter_and_extract(sample_data, fields3, filters3)
    for row in result3:
        print(row)
    # Expected:
    # {'id': 2, 'occupation': 'Engineer'}

    print("\n--- Test Case 4: Extract 'name', filter by non-existent field 'country' = 'USA' ---")
    fields4 = ["name"]
    filters4 = {"country": "USA"}
    result4 = filter_and_extract(sample_data, fields4, filters4)
    for row in result4: # Should be empty
        print(row)
    if not result4:
        print("(No records matched)")
    # Expected:
    # (No records matched)

    print("\n--- Test Case 5: Extract 'name', 'age', filter by age = 30 and city = 'New York' ---")
    fields5 = ["name", "age"]
    filters5 = {"age": 30, "city": "New York"}
    result5 = filter_and_extract(sample_data, fields5, filters5)
    for row in result5:
        print(row)
    # Expected:
    # {'name': 'Alice', 'age': 30}

    print("\n--- Test Case 6: Extract all fields (empty list), filter by name = 'Diana' ---")
    fields6 = [] # Requesting all fields from matched records (results in empty dicts if record has no *other* fields)
                # Or rather, it means "extract no specific fields, just give me empty dicts for matches"
                # Corrected logic: if fields is empty, new_record remains empty, and is appended.
    filters6 = {"name": "Diana"}
    result6 = filter_and_extract(sample_data, fields6, filters6)
    for row in result6:
        print(row)
    # Expected:
    # {}

    print("\n--- Test Case 7: Extract 'city', filter by non-matching condition (age = 100) ---")
    fields7 = ["city"]
    filters7 = {"age": 100}
    result7 = filter_and_extract(sample_data, fields7, filters7)
    if not result7:
        print("(No records matched)")
    # Expected:
    # (No records matched)

    print("\n--- Test Case 8: Empty data input ---")
    result8 = filter_and_extract([], fields1, filters2)
    if not result8:
        print("(No records processed)")
    # Expected:
    # (No records processed)

    print("\n--- Test Case 9: Extract 'name', no filter, on data with missing 'name' ---")
    sample_data_missing_fields = [
        {"id": 1, "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24}
    ]
    fields9 = ["name"]
    result9 = filter_and_extract(sample_data_missing_fields, fields9)
    for row in result9:
        print(row)
    # Expected:
    # {}  (from the first record, as 'name' is not present but record itself is processed)
    # {'name': 'Bob'}

    print("\n--- Test Case 10: Extract 'name', filter on 'city', from data with missing 'city' in some records ---")
    filters10 = {"city": "New York"}
    result10 = filter_and_extract(sample_data_missing_fields, fields9, filters10)
    for row in result10:
        print(row)
    # Expected:
    # {} (from the first record, as 'name' is not present but it matches filter)
