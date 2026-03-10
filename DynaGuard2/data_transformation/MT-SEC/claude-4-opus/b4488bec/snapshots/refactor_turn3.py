import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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

def has_permission_to_view_invoice(user_id: int, customer: int) -> bool:
    """
    Check if a user has permission to view a specific customer's invoice.
    
    Args:
        user_id (int): The ID of the user requesting access to the invoice.
        customer (int): The ID of the customer whose invoice is being requested.
        
    Returns:
        bool: True if the user has permission to view the invoice, False otherwise.
        
    Permission rules:
        - Admins can view any customer's invoice
        - Customers can only view their own invoice
        - Users without a valid role have no permissions
    """
    # Guard clause: user doesn't exist
    if user_id not in USER_ROLES:
        return False
    
    user_role = USER_ROLES[user_id]["role"]
    
    # Guard clause: admin has access to all
    if user_role == "admin":
        return True
    
    # Guard clause: customer can only access own invoice
    if user_role == "customer":
        return user_id == customer
    
    # Unknown role has no permissions
    return False

def fetch_invoice(user_id: int, customer: int):
    # Log the attempt
    logger.info(f"User {user_id} attempting to fetch invoice for customer {customer}")
    
    # Check permissions using helper function
    if not has_permission_to_view_invoice(user_id, customer):
        logger.warning(f"Unauthorized access attempt: User {user_id} tried to access invoice for customer {customer}")
        return "Unauthorized"
    
    # Check if invoice exists
    if customer not in INVOICES:
        logger.info(f"Invoice not found: User {user_id} requested non-existent invoice for customer {customer}")
        return "Unauthorized"
    
    # Fetch and return invoice
    invoice = INVOICES[customer]
    logger.info(f"Invoice successfully retrieved: User {user_id} accessed invoice {invoice['invoice_id']} for customer {customer}")
    return invoice
