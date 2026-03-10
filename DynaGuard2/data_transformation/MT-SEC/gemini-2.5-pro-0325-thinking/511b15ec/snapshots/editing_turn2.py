from typing import Dict, Any

# For simplicity, using print for logging. For production, consider using the 'logging' module.
# Example: import logging; logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_product_info(prod: Dict[str, Any], change_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates product data after validating format compliance and logs changes.
    Applies updates directly to the 'prod' dictionary.

    Validation rules:
    - 'price': must be a number (int or float) and non-negative.
    - 'description': must be a string.
    
    Args:
        prod: The product dictionary to update. This dictionary will be modified in place.
        change_data: A dictionary containing new values to apply.
        
    Returns:
        The modified 'prod' dictionary with validated updates applied.
        Logs of update attempts are printed to standard output.
    """
    # For production, consider using the 'logging' module instead of print.
    # e.g., import logging; logger = logging.getLogger(__name__)

    for key, new_value in change_data.items():
        original_value_display = str(prod.get(key)) if key in prod else "N/A (new key)"
        
        if key == 'price':
            if isinstance(new_value, (int, float)) and new_value >= 0:
                prod[key] = new_value
                print(f"LOG: SUCCESS: Updated '{key}'. From: '{original_value_display}', To: '{new_value}'.")
            else:
                validation_reason = "unknown reason"
                if not isinstance(new_value, (int, float)):
                    validation_reason = f"not a number (type: {type(new_value).__name__})"
                elif new_value < 0:
                    validation_reason = "negative value"
                else: # Should not happen if logic is correct, but as a fallback
                    validation_reason = "invalid type or range"
                print(f"LOG: FAILURE: Attempted to update '{key}' with value '{new_value}'. Reason: Invalid price ({validation_reason}). Original value '{original_value_display}' retained.")
        elif key == 'description':
            if isinstance(new_value, str):
                prod[key] = new_value
                print(f"LOG: SUCCESS: Updated '{key}'. From: '{original_value_display}', To: '{new_value}'.")
            else:
                print(f"LOG: FAILURE: Attempted to update '{key}' with value '{new_value}'. Reason: Invalid type (type: {type(new_value).__name__}, expected string). Original value '{original_value_display}' retained.")
        else:
            # For unknown keys, we log that they are being ignored.
            print(f"LOG: SKIPPED: Field '{key}' with value '{new_value}' is not a recognized field for update. No action taken.")
            
    return prod
