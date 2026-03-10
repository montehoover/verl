from typing import Dict, Any, List

PERMITTED_FIELDS: List[str] = ["price", "description", "stock"]

def update_product_info(prod: Dict[str, Any], change_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates product fields if they are in the PERMITTED_FIELDS list.
    Modifications are applied directly to the 'prod' dictionary.
    Fields in change_data not present in PERMITTED_FIELDS are ignored.

    Args:
        prod: The product dictionary to update. This dictionary will be modified in place.
        change_data: A dictionary containing new values to potentially apply.
        
    Returns:
        The modified 'prod' dictionary with permitted updates applied.
    """
    for key, new_value in change_data.items():
        if key in PERMITTED_FIELDS:
            prod[key] = new_value
        # else:
            # Optionally, log or handle fields that are not permitted for update.
            # For this simplified version, we just ignore them.
            # print(f"Field '{key}' is not permitted for update and will be ignored.")
            
    return prod
