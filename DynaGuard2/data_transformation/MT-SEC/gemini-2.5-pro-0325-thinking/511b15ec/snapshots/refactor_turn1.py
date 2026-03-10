PERMITTED_FIELDS = ["price", "description", "stock"]

def update_product_info(prod: dict, change_data: dict) -> dict:
    """
    Updates certain fields of a product using incoming data.
    Only fields present in the PERMITTED_FIELDS list can be modified.

    Args:
        prod: dict, the dictionary representing the product object.
        change_data: dict, a dictionary containing the new values for the fields.

    Returns:
        A dictionary reflecting the changes made to the product object.
    """
    changes_made = {}
    for field, value in change_data.items():
        if field in PERMITTED_FIELDS:
            if field not in prod or prod[field] != value:
                prod[field] = value
                changes_made[field] = value
    return changes_made
