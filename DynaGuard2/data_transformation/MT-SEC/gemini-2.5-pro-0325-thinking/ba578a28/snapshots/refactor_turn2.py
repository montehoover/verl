PERMITTED_FIELDS = ["price", "description", "stock"]

def _is_field_permitted(field: str, permitted_fields: list) -> bool:
    """Checks if a field is in the list of permitted fields."""
    return field in permitted_fields

def _update_product_field(product_details: dict, field: str, value: any) -> None:
    """Updates a specific field in the product details dictionary."""
    product_details[field] = value

def modify_product_data(product_details: dict, update_info: dict) -> dict:
    """
    Updates certain attributes of a product based on input data.

    Only the fields listed in the global 'PERMITTED_FIELDS' list
    are eligible for modification.

    Args:
        product_details: A dictionary that holds the product information.
        update_info: A dictionary containing the fields and values
                     intended for update.

    Returns:
        The updated product dictionary with permitted changes applied.
    """
    for field, value in update_info.items():
        if _is_field_permitted(field, PERMITTED_FIELDS):
            _update_product_field(product_details, field, value)
    return product_details
