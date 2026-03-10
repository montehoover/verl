from typing import Dict, Any

def update_product_info(prod: Dict[str, Any], change_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks product data for format compliance.
    Validates values in change_data against specified formats:
    - 'price' must be a number (int or float).
    - 'description' must be a string.
    
    Args:
        prod: The existing product dictionary (currently unused but part of the signature).
        change_data: A dictionary containing new values to potentially update.
        
    Returns:
        A dictionary containing only the validated key-value pairs from change_data.
        Invalid fields or fields without specified validation rules are omitted.
    """
    validated_updates: Dict[str, Any] = {}
    for key, value in change_data.items():
        if key == 'price':
            if isinstance(value, (int, float)):
                validated_updates[key] = value
            else:
                # Invalid type for price, skipping.
                # Consider logging or raising an error for production code.
                pass
        elif key == 'description':
            if isinstance(value, str):
                validated_updates[key] = value
            else:
                # Invalid type for description, skipping.
                # Consider logging or raising an error for production code.
                pass
        # Other keys in change_data are currently ignored if they don't have specific validation rules.
        # If strict checking for only known/allowed keys is required, 
        # an error could be raised for unknown keys in an 'else' block here.
        
    return validated_updates
