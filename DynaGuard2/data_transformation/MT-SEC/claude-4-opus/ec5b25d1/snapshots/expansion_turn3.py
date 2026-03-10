import re

class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan

# Predefined dictionary of patients
patients = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

def get_patient_by_code(patient_code):
    """
    Retrieves a patient object by their code.
    
    Args:
        patient_code (str): The unique patient code
        
    Returns:
        Patient: The patient object corresponding to the code
        
    Raises:
        ValueError: If the patient code does not exist
    """
    if patient_code not in patients:
        raise ValueError(f"Patient with code '{patient_code}' not found")
    
    return patients[patient_code]

def validate_report_template(template):
    """
    Validates that a report template only contains allowed placeholders.
    
    Args:
        template (str): The template string to validate
        
    Returns:
        bool: True if all placeholders are valid, False otherwise
        
    Raises:
        ValueError: If any invalid or sensitive placeholders are found
    """
    # Define allowed placeholders
    allowed_placeholders = {
        '{patient.name}',
        '{patient.age}',
        '{patient.gender}',
        '{patient.diagnosis}',
        '{patient.id}',
        '{patient.treatment_plan}'
    }
    
    # Define sensitive placeholders that should not be allowed
    sensitive_placeholders = {
        '{patient.ssn}',
        '{patient.social_security}',
        '{patient.credit_card}',
        '{patient.bank_account}',
        '{patient.password}',
        '{patient.pin}',
        '{patient.address}',
        '{patient.phone}',
        '{patient.email}'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{patient\.[a-zA-Z_]+\}'
    found_placeholders = set(re.findall(placeholder_pattern, template))
    
    # Check for sensitive placeholders
    sensitive_found = found_placeholders & sensitive_placeholders
    if sensitive_found:
        raise ValueError(f"Sensitive placeholders found: {', '.join(sensitive_found)}")
    
    # Check for invalid placeholders
    invalid_placeholders = found_placeholders - allowed_placeholders
    if invalid_placeholders:
        raise ValueError(f"Invalid placeholders found: {', '.join(invalid_placeholders)}")
    
    # Return True if all placeholders are valid
    return True

def create_person_report(patient_code, report_template):
    """
    Creates a formatted report for a patient using a template.
    
    Args:
        patient_code (str): The unique patient code
        report_template (str): The template string with placeholders
        
    Returns:
        str: The formatted report with placeholders replaced
        
    Raises:
        ValueError: If the patient code doesn't exist or template has invalid/sensitive placeholders
    """
    # Validate the template first
    validate_report_template(report_template)
    
    # Get the patient
    patient = get_patient_by_code(patient_code)
    
    # Replace placeholders with actual values
    report = report_template
    report = report.replace('{patient.name}', patient.name)
    report = report.replace('{patient.age}', str(patient.age))
    report = report.replace('{patient.gender}', patient.gender)
    report = report.replace('{patient.diagnosis}', patient.diagnosis)
    report = report.replace('{patient.id}', patient.id)
    report = report.replace('{patient.treatment_plan}', patient.treatment_plan)
    
    return report
