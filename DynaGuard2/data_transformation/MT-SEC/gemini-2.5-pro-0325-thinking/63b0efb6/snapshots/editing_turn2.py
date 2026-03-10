def extract_fields(data: list[dict], fields: list[str], conditions: dict) -> list[dict]:
    """
    Extracts specified fields from a list of dictionaries and filters records based on conditions.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.
        conditions: A dictionary specifying field conditions to filter records.
                    Example: {"age": 30, "city": "New York"}

    Returns:
        A new list of dictionaries, where each dictionary contains only
        the specified fields from the original dictionaries that match all conditions.

    Raises:
        ValueError: If a condition field is not found in a record when checking conditions.
    """
    result = []
    for record in data:
        record_matches_conditions = True
        # Apply conditions if any are provided (i.e., if conditions dictionary is not empty)
        if conditions: 
            for cond_key, cond_value in conditions.items():
                if cond_key not in record:
                    raise ValueError(f"Condition field '{cond_key}' not found in record: {record}")
                if record[cond_key] != cond_value:
                    record_matches_conditions = False
                    break  # Stop checking other conditions for this record

        if record_matches_conditions:
            # If record matches all conditions (or if conditions was empty), extract fields
            new_record = {}
            for field in fields:
                if field in record:
                    new_record[field] = record[field]
            result.append(new_record) # Add the extracted record (could be empty if fields is empty or no specified fields found)
    return result

if __name__ == '__main__':
    # Example Usage
    sample_data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer"},
        {"id": 3, "name": "Charlie", "city": "London", "age": 35}
    ]

    # --- Original examples (adapted with empty conditions dictionary {}) ---
    print("--- Original behavior with empty conditions ---")
    fields_to_extract = ["name", "age"]
    # Pass empty dict for conditions to signify no filtering
    extracted_data = extract_fields(sample_data, fields_to_extract, {})
    print("Extracted data (name, age) with no filter:")
    for item in extracted_data:
        print(item)

    fields_to_extract_2 = ["id", "city", "occupation"]
    extracted_data_2 = extract_fields(sample_data, fields_to_extract_2, {})
    print("\nExtracted data (id, city, occupation) with no filter:")
    for item in extracted_data_2:
        print(item)

    fields_to_extract_3 = ["non_existent_field"]
    extracted_data_3 = extract_fields(sample_data, fields_to_extract_3, {})
    print("\nExtracted data (non_existent_field) with no filter (expect list of empty dicts):")
    for item in extracted_data_3:
        print(item) 

    fields_to_extract_4 = [] # Empty list of fields
    extracted_data_4 = extract_fields(sample_data, fields_to_extract_4, {})
    print("\nExtracted data (no fields) with no filter (expect list of empty dicts):")
    for item in extracted_data_4:
        print(item)

    empty_data = []
    extracted_data_empty = extract_fields(empty_data, fields_to_extract, {})
    print("\nExtracted data (empty input data) with no filter:")
    for item in extracted_data_empty:
        print(item)

    # --- New examples with filtering ---
    print("\n--- New examples with filtering ---")

    # Example 1: Filter by age
    conditions_age_30 = {"age": 30}
    extracted_filtered_1 = extract_fields(sample_data, ["name", "city"], conditions_age_30)
    print("\nFiltered by age=30, extract (name, city):")
    for item in extracted_filtered_1:
        print(item) # Expected: [{'name': 'Alice', 'city': 'New York'}]

    # Example 2: Filter by multiple fields
    conditions_bob = {"name": "Bob", "occupation": "Engineer"}
    extracted_filtered_2 = extract_fields(sample_data, ["id", "age"], conditions_bob)
    print("\nFiltered by name='Bob' AND occupation='Engineer', extract (id, age):")
    for item in extracted_filtered_2:
        print(item) # Expected: [{'id': 2, 'age': 24}]

    # Example 3: Filter resulting in no matches (condition value mismatch)
    conditions_no_match_value = {"city": "Paris"}
    extracted_filtered_3 = extract_fields(sample_data, ["name"], conditions_no_match_value)
    print("\nFiltered by city='Paris', extract (name):")
    for item in extracted_filtered_3:
        print(item) # Expected: []

    # Example 4: Filter with empty fields list
    extracted_filtered_4 = extract_fields(sample_data, [], conditions_age_30)
    print("\nFiltered by age=30, extract (no fields) (expect list with one empty dict):")
    for item in extracted_filtered_4:
        print(item) # Expected: [{}]

    # Example 5: Filter with condition on a non-existent field (ValueError)
    print("\nAttempting filter with condition on non-existent field 'country':")
    conditions_bad_field = {"country": "USA"}
    try:
        extract_fields(sample_data, ["name"], conditions_bad_field)
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")

    # Example 6: Filter with conditions that don't match any record's value for a field
    conditions_age_100 = {"age": 100}
    extracted_filtered_5 = extract_fields(sample_data, ["name"], conditions_age_100)
    print("\nFiltered by age=100, extract (name):")
    for item in extracted_filtered_5:
        print(item) # Expected: []

    # Example 7: All records match condition (empty condition dict), 
    # but some fields to extract are not present in all matching records
    conditions_all_match_implicit = {} 
    fields_mixed_presence = ["name", "occupation"]
    extracted_filtered_6 = extract_fields(sample_data, fields_mixed_presence, conditions_all_match_implicit)
    print("\nNo filter, extract (name, occupation):")
    for item in extracted_filtered_6:
        print(item)
