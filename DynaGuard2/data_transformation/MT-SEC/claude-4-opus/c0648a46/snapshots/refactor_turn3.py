import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_profile(worker, changes, is_privileged):
    # Create a copy of the worker dictionary to avoid modifying the original
    updated_worker = worker.copy()
    
    # Log the start of profile modification
    logger.info(f"Starting profile modification for employee (privileged={is_privileged})")
    
    for field, value in changes.items():
        # Skip if field doesn't exist in worker profile
        if field not in updated_worker:
            logger.warning(f"Attempted to modify non-existent field '{field}'")
            continue
            
        # Skip if non-privileged user tries to edit restricted field
        if not is_privileged and field not in EDITABLE_FIELDS:
            logger.warning(f"Non-privileged user attempted to modify restricted field '{field}'")
            continue
            
        # Log successful field update
        old_value = updated_worker[field]
        updated_worker[field] = value
        logger.info(f"Successfully updated field '{field}' from '{old_value}' to '{value}'")
    
    # Log completion
    logger.info("Profile modification completed")
    
    return updated_worker
