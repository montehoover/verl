import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
}

INVOICES = {
    2: {"invoice_id": 1001, "amount": "$200", "details": "Invoice 1001 for User 2"},
    3: {"invoice_id": 1002, "amount": "$150", "details": "Invoice 1002 for User 3"},
    4: {"invoice_id": 1003, "amount": "$300", "details": "Invoice 1003 for User 4"}
}

def _is_authorized(current_user_id: int, customer_id: int, user_roles_data: dict) -> bool:
    """
    Checks if the current user is authorized to access the customer's invoice.
    Admins can access any invoice. Customers can only access their own.
    """
    current_user_data = user_roles_data.get(current_user_id)
    if not current_user_data:
        return False  # Current user not found

    current_user_role = current_user_data.get("role")

    if current_user_role == "admin":
        return True
    
    if current_user_role == "customer":
        return current_user_id == customer_id
            
    return False # Unknown role or other cases

def _get_invoice_data(customer_id: int, invoices_data: dict):
    """
    Retrieves invoice data for a given customer_id.
    Returns invoice data or 'Invoice not found'.
    """
    invoice_data = invoices_data.get(customer_id)
    return invoice_data if invoice_data else 'Invoice not found'

def download_invoice(current_user_id: int, customer_id: int):
    """
    Allows customers to download their own invoices and administrators
    to download any customer's invoice.

    Args:
        current_user_id: The user_id of the actor requesting to download the invoice.
        customer_id: The user_id of the customer whose invoice is requested.

    Returns:
        The invoice data if the download is authorized, else 'Unauthorized'.
    """
    logging.info(f"Invoice download request: current_user_id={current_user_id}, customer_id={customer_id}")

    authorized = _is_authorized(current_user_id, customer_id, USER_ROLES)

    if not authorized:
        logging.warning(f"Unauthorized invoice download attempt: current_user_id={current_user_id}, customer_id={customer_id}")
        return 'Unauthorized'
    
    logging.info(f"Authorization successful for invoice download: current_user_id={current_user_id}, customer_id={customer_id}")
    
    invoice_data = _get_invoice_data(customer_id, INVOICES)
    
    if invoice_data == 'Invoice not found':
        logging.warning(f"Invoice not found for customer_id={customer_id} (requested by current_user_id={current_user_id})")
    else:
        logging.info(f"Invoice data retrieved for customer_id={customer_id} (requested by current_user_id={current_user_id})")
        
    return invoice_data
