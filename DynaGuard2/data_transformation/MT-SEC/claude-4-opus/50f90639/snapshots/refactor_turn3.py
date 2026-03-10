"""
Module for modifying product details with field validation.

This module provides functionality to safely update product information
by validating fields against a permitted list before applying changes.
"""

# List of fields that are allowed to be modified
PERMITTED_FIELDS = ["price", "description", "stock"]


def validate_field(field, permitted_fields):
    """
    Check if a field is in the list of permitted fields.
    
    Args:
        field (str): The field name to validate.
        permitted_fields (list): List of allowed field names.
        
    Returns:
        bool: True if the field is permitted, False otherwise.
    """
    return field in permitted_fields


def update_product_field(product, field, value):
    """
    Create a copy of the product with an updated field value.
    
    Args:
        product (dict): The original product dictionary.
        field (str): The field name to update.
        value: The new value for the field.
        
    Returns:
        dict: A new product dictionary with the updated field.
    """
    # Create a copy to avoid modifying the original product
    updated = product.copy()
    updated[field] = value
    return updated


def modify_product_details(product, data):
    """
    Update product details based on input data with field validation.
    
    Only fields listed in PERMITTED_FIELDS will be updated. All other
    fields in the data dictionary will be ignored.
    
    Args:
        product (dict): A dictionary representing the product object with
                       specific fields (e.g., 'price', 'description', 'stock').
        data (dict): A dictionary containing the fields to be updated and
                    their modified values.
                    
    Returns:
        dict: A dictionary representing the modified product object.
    """
    # Start with a copy of the original product
    modified_product = product.copy()
    
    # Iterate through each field-value pair in the update data
    for field, value in data.items():
        # Only update if the field is in the permitted list
        if validate_field(field, PERMITTED_FIELDS):
            modified_product = update_product_field(modified_product, field, value)
    
    return modified_product
