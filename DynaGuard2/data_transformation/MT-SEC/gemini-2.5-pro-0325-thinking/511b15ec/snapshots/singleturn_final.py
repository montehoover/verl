PERMITTED_FIELDS = ["price", "description", "stock"]

def update_product_info(prod: dict, change_data: dict) -> dict:
    """
    Updates certain fields of a product using incoming data.
    Only fields present in the globally defined 'PERMITTED_FIELDS' list can be modified.

    Args:
        prod: dict, the dictionary representing the product object with its corresponding fields.
        change_data: dict, a dictionary containing the new values for the fields that need to be updated.

    Returns:
        A dictionary reflecting the changes made to the product object.
    """
    updated_prod = prod.copy()  # Work on a copy to avoid modifying the original dict directly if it's not desired
    for key, value in change_data.items():
        if key in PERMITTED_FIELDS:
            updated_prod[key] = value
    return updated_prod
