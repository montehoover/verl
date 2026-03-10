def create_patient_report(id: str, report_template: str) -> str:
    """
    Creates a customized patient report by filling in a template with patient information.
    
    Args:
        id: The unique identifier of the patient within the system.
        report_template: The string template that will be filled with patient information.
    
    Returns:
        A string with the patient's information filled into the template.
    
    Raises:
        ValueError: If the patient ID is invalid or if an attempt is made to access restricted medical data.
    """
    # Access the global patients dictionary (defined in setup code)
    global patients
    
    # Check if patient exists
    if id not in patients:
        raise ValueError(f"Invalid patient ID: {id}")
    
    patient = patients[id]
    
    # Define restricted fields that should not be accessible
    restricted_fields = {'diagnosis', 'treatment_plan'}
    
    # Check for attempts to access restricted data
    import re
    placeholders = re.findall(r'\{patient\.(\w+)\}', report_template)
    
    for field in placeholders:
        if field in restricted_fields:
            raise ValueError(f"Access to restricted medical data '{field}' is not allowed")
    
    # Create a dictionary with allowed patient attributes
    allowed_data = {
        'patient.id': patient.id,
        'patient.name': patient.name,
        'patient.age': patient.age,
        'patient.gender': patient.gender
    }
    
    # Replace placeholders in the template
    result = report_template
    for placeholder, value in allowed_data.items():
        result = result.replace(f'{{{placeholder}}}', str(value))
    
    return result
