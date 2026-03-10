import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EDITABLE_FIELDS = {"name", "position", "department"}

def validate_modifications(modifications, is_superuser):
    """Filter modifications to only include allowed fields."""
    if is_superuser:
        logger.info(f"Superuser attempting to modify fields: {list(modifications.keys())}")
        return modifications
    
    allowed_modifications = {field: value for field, value in modifications.items() 
                           if field in EDITABLE_FIELDS}
    
    restricted_fields = set(modifications.keys()) - EDITABLE_FIELDS
    if restricted_fields:
        logger.warning(f"Non-admin user attempted to modify restricted fields: {list(restricted_fields)}")
    
    logger.info(f"Non-admin user modifying allowed fields: {list(allowed_modifications.keys())}")
    return allowed_modifications

def apply_modifications(profile, validated_modifications):
    """Apply validated modifications to a copy of the profile."""
    if not validated_modifications:
        logger.info("No valid modifications to apply")
        return profile.copy()
    
    updated_profile = profile.copy()
    updated_profile.update(validated_modifications)
    
    changed_fields = []
    for field, new_value in validated_modifications.items():
        old_value = profile.get(field, "N/A")
        changed_fields.append(f"{field}: '{old_value}' -> '{new_value}'")
    
    logger.info(f"Profile updated successfully. Changes: {', '.join(changed_fields)}")
    return updated_profile

def edit_profile_data(profile, modifications, is_superuser):
    user_type = "superuser" if is_superuser else "non-admin"
    logger.info(f"Profile edit initiated by {user_type} for employee: {profile.get('name', 'Unknown')}")
    
    validated_mods = validate_modifications(modifications, is_superuser)
    return apply_modifications(profile, validated_mods)
