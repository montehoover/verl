import logging
import copy # For deep copying product_details for logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    original_product_details = copy.deepcopy(product_details)
    logging.info(f"Original product details: {original_product_details}")
    logging.info(f"Update info: {update_info}")

    updated_fields_count = 0
    for field, value in update_info.items():
        if _is_field_permitted(field, PERMITTED_FIELDS):
            old_value = product_details.get(field)
            _update_product_field(product_details, field, value)
            logging.info(f"Updated field '{field}': from '{old_value}' to '{value}'")
            updated_fields_count += 1
        else:
            logging.warning(f"Attempt to update non-permitted field '{field}' was ignored.")

    if updated_fields_count > 0:
        logging.info(f"Final product details after update: {product_details}")
    else:
        logging.info("No fields were updated as per permitted fields or update info.")
    return product_details
