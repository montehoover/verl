def extract_fields(data: list[dict], fields: list[str], conditions: dict = None) -> list[dict]:
    """
    Extracts specified fields from a list of dictionaries after filtering.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.
        conditions: A dictionary of conditions to filter records.
                    Only records that meet all conditions are processed.
                    Defaults to None (no filtering).

    Returns:
        A new list of dictionaries, where each dictionary contains only the
        specified fields from the corresponding, filtered dictionary in the input data.
        If a field is not present in an input dictionary, it will be omitted
        from the output dictionary for that record.

    Raises:
        ValueError: If a field specified in `conditions` is not present in a record
                    being evaluated.
    """
    result = []
    for record in data:
        # Apply filtering based on conditions
        if conditions:
            record_matches_all_conditions = True
            for cond_key, cond_value in conditions.items():
                if cond_key not in record:
                    raise ValueError(f"Condition field '{cond_key}' not found in record: {record}")
                if record[cond_key] != cond_value:
                    record_matches_all_conditions = False
                    break
            
            if not record_matches_all_conditions:
                continue  # Skip to the next record if conditions are not met

        # If record passes filters (or no filters), proceed with field extraction
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        
        if new_record:  # Add record only if it's not empty after extraction
            result.append(new_record)
    return result

if __name__ == '__main__':
    # Example Usage
    sample_data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer"},
        {"id": 3, "name": "Charlie", "city": "London", "country": "UK"},
        {"id": 4, "name": "Diana"} # Record with no fields to extract if fields are e.g. ["age", "city"]
    ]

    fields_to_extract = ["name", "city"]
    extracted_data = extract_fields(sample_data, fields_to_extract)
    print("Original Data:")
    for item in sample_data:
        print(item)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract))
    for item in extracted_data:
        print(item)

    fields_to_extract_2 = ["age", "occupation"]
    extracted_data_2 = extract_fields(sample_data, fields_to_extract_2)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract_2))
    for item in extracted_data_2:
        print(item)

    fields_to_extract_3 = ["country"]
    extracted_data_3 = extract_fields(sample_data, fields_to_extract_3)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract_3))
    for item in extracted_data_3:
        print(item)

    # Example with no matching fields in any record
    fields_to_extract_4 = ["non_existent_field"]
    extracted_data_4 = extract_fields(sample_data, fields_to_extract_4)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract_4))
    for item in extracted_data_4:
        print(item)
    if not extracted_data_4:
        print("No data extracted as fields were not found.")

    print("\n--- Testing with filtering conditions ---")

    # Example with filtering: extract name and age for records where city is "New York"
    fields_to_extract_filtered = ["name", "age"]
    conditions_filter_1 = {"city": "New York"}
    print(f"\nExtracting fields {fields_to_extract_filtered} with conditions {conditions_filter_1}:")
    try:
        filtered_data_1 = extract_fields(sample_data, fields_to_extract_filtered, conditions_filter_1)
        for item in filtered_data_1:
            print(item) # Expected: {"name": "Alice", "age": 30}
    except ValueError as e:
        print(f"Error: {e}")

    # Example with filtering: extract name, city and occupation for records where age is 24
    fields_to_extract_filtered_2 = ["name", "city", "occupation"]
    conditions_filter_2 = {"age": 24}
    print(f"\nExtracting fields {fields_to_extract_filtered_2} with conditions {conditions_filter_2}:")
    try:
        filtered_data_2 = extract_fields(sample_data, fields_to_extract_filtered_2, conditions_filter_2)
        for item in filtered_data_2:
            # Expected: {"name": "Bob", "occupation": "Engineer"} (city is not in Bob's record, so not extracted)
            print(item) 
    except ValueError as e:
        print(f"Error: {e}")

    # Example with filtering: no records match the conditions
    conditions_filter_3 = {"city": "Paris"}
    # Using fields_to_extract from a previous example: ["name", "city"]
    print(f"\nExtracting fields {fields_to_extract} with conditions {conditions_filter_3}:")
    try:
        filtered_data_3 = extract_fields(sample_data, fields_to_extract, conditions_filter_3)
        if filtered_data_3:
            for item in filtered_data_3:
                print(item)
        else:
            print("No records matched the conditions.")
    except ValueError as e:
        print(f"Error: {e}")

    # Example with filtering: record matches conditions, but specified fields for extraction are not in the record
    fields_empty_extraction = ["height"] # Assuming 'height' is not a common field in sample_data
    conditions_alice = {"name": "Alice"}
    print(f"\nExtracting fields {fields_empty_extraction} with conditions {conditions_alice}:")
    try:
        filtered_data_empty = extract_fields(sample_data, fields_empty_extraction, conditions_alice)
        if filtered_data_empty:
            for item in filtered_data_empty:
                print(item)
        else:
            # This is expected if Alice's record is found but 'height' is not in it.
            print("Record matched conditions, but no specified fields found for extraction.") 
    except ValueError as e:
        print(f"Error: {e}")

    # Example that should raise ValueError due to condition field not in a record
    print("\nTesting ValueError for condition on non-existent field:")
    conditions_error = {"non_existent_key": "value"}
    fields_for_error_test = ["name"]
    try:
        print(f"Attempting to extract fields {fields_for_error_test} with conditions {conditions_error}:")
        # This call should raise ValueError because 'non_existent_key' is not in the first record.
        extract_fields(sample_data, fields_for_error_test, conditions_error)
        print("ValueError was not raised as expected.") # Should not be reached
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")

    # Example: conditions is None (should behave as original, filtering only by fields)
    print("\nTesting with conditions=None (should extract based on fields only):")
    # Using fields_to_extract from a previous example: ["name", "city"]
    extracted_data_no_conditions = extract_fields(sample_data, fields_to_extract, None)
    for item in extracted_data_no_conditions:
        print(item)

    # Example: conditions is an empty dict (should behave as original, filtering only by fields)
    print("\nTesting with conditions={{}} (should extract based on fields only):")
    extracted_data_empty_conditions = extract_fields(sample_data, fields_to_extract, {})
    for item in extracted_data_empty_conditions:
        print(item)
