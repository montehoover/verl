import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def is_authorized_to_download(current_user_id, customer_id):
    """Check if the current user is authorized to download the customer's invoice."""
    if current_user_id not in USER_ROLES:
        return False
    
    user_role = USER_ROLES[current_user_id]["role"]
    
    if user_role == "admin":
        return True
    elif user_role == "customer":
        return current_user_id == customer_id
    else:
        return False

def get_invoice(customer_id):
    """Retrieve the invoice for the given customer."""
    return INVOICES.get(customer_id, None)

def download_invoice(current_user_id, customer_id):
    logger.info(f"Invoice download request: current_user_id={current_user_id}, customer_id={customer_id}")
    
    if not is_authorized_to_download(current_user_id, customer_id):
        logger.warning(f"Unauthorized invoice download attempt: current_user_id={current_user_id}, customer_id={customer_id}")
        return "Unauthorized"
    
    invoice = get_invoice(customer_id)
    if invoice is None:
        logger.warning(f"Invoice not found: current_user_id={current_user_id}, customer_id={customer_id}")
        return "Unauthorized"
    
    logger.info(f"Invoice download successful: current_user_id={current_user_id}, customer_id={customer_id}, invoice_id={invoice['invoice_id']}")
    return invoice
