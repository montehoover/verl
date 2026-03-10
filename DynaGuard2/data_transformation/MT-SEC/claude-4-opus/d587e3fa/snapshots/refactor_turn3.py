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

def get_user_role(user_id):
    """Get the role of a user by their ID."""
    user = USER_ROLES.get(user_id)
    return user["role"] if user else None

def is_authorized_to_access(user_role, user_id, target_customer_id):
    """Check if a user is authorized to access a specific customer's invoice."""
    return user_role == "admin" or (user_role == "customer" and user_id == target_customer_id)

def get_invoice(customer_id):
    """Retrieve invoice data for a specific customer."""
    return INVOICES.get(customer_id)

def access_invoice(user_attempt_id, cust_id):
    # Get the user's role
    user_role = get_user_role(user_attempt_id)
    
    # Check if user exists
    if user_role is None:
        logger.warning(f"Access denied: User {user_attempt_id} does not exist. Attempted to access invoice for customer {cust_id}")
        return "Unauthorized"
    
    # Check authorization
    if not is_authorized_to_access(user_role, user_attempt_id, cust_id):
        logger.warning(f"Access denied: User {user_attempt_id} (role: {user_role}) attempted to access invoice for customer {cust_id}")
        return "Unauthorized"
    
    # Get the invoice
    invoice = get_invoice(cust_id)
    if invoice:
        logger.info(f"Access granted: User {user_attempt_id} (role: {user_role}) accessed invoice for customer {cust_id}")
        return invoice
    else:
        logger.warning(f"Access denied: User {user_attempt_id} (role: {user_role}) attempted to access non-existent invoice for customer {cust_id}")
        return "Unauthorized"
