# Permitted fields that can be modified
PERMITTED_FIELDS = ["price", "description", "stock"]

def modify_product_data(product_details: dict, update_info: dict) -> dict:
    """
    Update allowed fields of a product based on provided update information.

    Args:
        product_details (dict): A dictionary holding the current product information.
        update_info (dict): A dictionary containing fields and values intended for update.

    Returns:
        dict: The updated product dictionary with permitted changes applied.
    """
    # Ensure input dictionaries are valid; default to empty to avoid TypeErrors
    product_details = product_details or {}
    update_info = update_info or {}

    # Create a shallow copy to avoid mutating the original input
    updated = dict(product_details)

    # Apply only permitted updates
    for key, value in update_info.items():
        if key in PERMITTED_FIELDS:
            updated[key] = value

    return updated
