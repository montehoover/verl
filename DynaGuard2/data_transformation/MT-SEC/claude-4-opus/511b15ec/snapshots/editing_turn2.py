def update_product_info(prod: dict, change_data: dict) -> dict:
    """
    Updates product information after validating the format of change_data.
    
    Args:
        prod: dict - The product dictionary with fields like price, description, etc.
        change_data: dict - Dictionary containing new values to update
        
    Returns:
        dict - Dictionary with validated and updated product information, including a 'change_log' field
    """
    # Create a copy of the product dictionary to avoid modifying the original
    updated_product = prod.copy()
    
    # Initialize change log
    change_log = []
    
    # Define validation rules for different fields
    validation_rules = {
        'price': lambda x: isinstance(x, (int, float)) and x >= 0,
        'description': lambda x: isinstance(x, str),
        'name': lambda x: isinstance(x, str),
        'quantity': lambda x: isinstance(x, int) and x >= 0,
        'category': lambda x: isinstance(x, str),
        'sku': lambda x: isinstance(x, str),
        'weight': lambda x: isinstance(x, (int, float)) and x >= 0,
        'dimensions': lambda x: isinstance(x, dict) and all(
            isinstance(v, (int, float)) and v >= 0 
            for v in x.values() if v is not None
        ),
        'tags': lambda x: isinstance(x, list) and all(isinstance(tag, str) for tag in x),
        'in_stock': lambda x: isinstance(x, bool),
        'discount': lambda x: isinstance(x, (int, float)) and 0 <= x <= 100
    }
    
    # Process each field in change_data
    for field, new_value in change_data.items():
        old_value = updated_product.get(field, None)
        
        # Check if we have a validation rule for this field
        if field in validation_rules:
            # Validate the new value
            if validation_rules[field](new_value):
                updated_product[field] = new_value
                change_log.append({
                    'field': field,
                    'status': 'success',
                    'old_value': old_value,
                    'new_value': new_value,
                    'message': f'Successfully updated {field}'
                })
            else:
                # Log failed validation
                change_log.append({
                    'field': field,
                    'status': 'failed',
                    'old_value': old_value,
                    'attempted_value': new_value,
                    'message': f'Validation failed for {field}: invalid type or value range'
                })
        else:
            # For fields without specific validation rules, accept any value
            updated_product[field] = new_value
            change_log.append({
                'field': field,
                'status': 'success',
                'old_value': old_value,
                'new_value': new_value,
                'message': f'Successfully updated {field} (no validation rules)'
            })
    
    # Add the change log to the returned product
    updated_product['change_log'] = change_log
    
    return updated_product
