PERMITTED_FIELDS = ["price", "description", "stock"]

def update_item_information(product_info: dict, new_data: dict) -> dict:
    """
    Modifies product attributes based on new data, restricted to permitted fields.

    Args:
        product_info: A dictionary instance representing a product.
        new_data: A dictionary containing key-value pairs for intended updates.

    Returns:
        A dictionary object reflecting the updated product state.
    """
    updated_product_info = product_info.copy()  # Work on a copy

    for key, value in new_data.items():
        if key in PERMITTED_FIELDS:
            updated_product_info[key] = value
    
    return updated_product_info

if __name__ == '__main__':
    # Example Usage from the problem description
    product_info_example = {
      "price": 100,
      "description": "Basic Widget",
      "stock": 50,
      "category": "tools"
    }
    new_data_example = {
      "price": 120,
      "description": "Upgraded Widget V2", # This should be updated
      "stock": 45, # This should be updated
      "color": "red" # This should be ignored
    }

    expected_output_example = {
      "price": 120,
      "description": "Upgraded Widget V2",
      "stock": 45,
      "category": "tools"
    }
    
    print(f"Input Product Info: {product_info_example}")
    print(f"Input New Data: {new_data_example}")
    
    output = update_item_information(product_info_example, new_data_example)
    print(f"Output Product Info: {output}")
    
    # Verify against the example output structure (ignoring category as it's not in PERMITTED_FIELDS for update)
    # The example output provided in the prompt seems to only update price.
    # Let's test with a more comprehensive example based on PERMITTED_FIELDS.
    
    product_info_test_2 = {
      "price": 100,
      "description": "Basic Widget",
      "stock": 50,
      "category": "tools"
    }
    new_data_test_2 = {
      "price": 120, # Permitted
      "description": "An even better widget", # Permitted
      "stock": 40, # Permitted
      "category": "premium tools", # Not permitted
      "manufacturer": "WidgetCorp" # Not permitted
    }
    
    print("\n--- More Comprehensive Test ---")
    print(f"Input Product Info: {product_info_test_2}")
    print(f"Input New Data: {new_data_test_2}")
    
    output_test_2 = update_item_information(product_info_test_2, new_data_test_2)
    print(f"Output Product Info: {output_test_2}")
    
    expected_output_test_2 = {
      "price": 120,
      "description": "An even better widget",
      "stock": 40,
      "category": "tools" # Unchanged
    }
    print(f"Expected Output: {expected_output_test_2}")
    assert output_test_2 == expected_output_test_2, "Comprehensive test failed!"
    print("Comprehensive test passed!")

    # Test with the exact example from the prompt
    product_info_prompt_example = {
      "price": 100,
      "description": "Basic Widget",
      "stock": 50,
      "category": "tools"
    }
    new_data_prompt_example = {
      "price": 120
    }
    output_prompt_example = update_item_information(product_info_prompt_example, new_data_prompt_example)
    expected_output_prompt_example = {
      "price": 120,
      "description": "Basic Widget",
      "stock": 50,
      "category": "tools"
    }
    print("\n--- Prompt Example Test ---")
    print(f"Input Product Info: {product_info_prompt_example}")
    print(f"Input New Data: {new_data_prompt_example}")
    print(f"Output Product Info: {output_prompt_example}")
    print(f"Expected Output: {expected_output_prompt_example}")
    assert output_prompt_example == expected_output_prompt_example, "Prompt example test failed!"
    print("Prompt example test passed!")
