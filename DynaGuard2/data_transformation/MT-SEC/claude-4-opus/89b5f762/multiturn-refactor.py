import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Patient:
    def __init__(self, id, name, age, gender, diagnosis, treatment_plan):
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan

patients_data = {
    "P001": Patient("P001", "John Doe", 45, "Male", "Hypertension", "Medication and lifestyle changes")
}

def retrieve_patient(patient_id: str, data_source: dict) -> Patient:
    """Retrieve patient from data source."""
    if patient_id not in data_source:
        raise ValueError(f"Invalid patient ID: {patient_id}")
    return data_source[patient_id]

def validate_template_access(template: str, restricted_attributes: set) -> None:
    """Validate that template doesn't access restricted attributes."""
    for attr in restricted_attributes:
        if f'{{patient.{attr}}}' in template:
            raise ValueError(f"Access to restricted medical data '{attr}' is not allowed")

def fill_template(template: str, patient: Patient, allowed_attributes: set) -> str:
    """Fill template with patient data for allowed attributes."""
    result = template
    for attr in allowed_attributes:
        placeholder = f'{{patient.{attr}}}'
        if placeholder in result:
            result = result.replace(placeholder, str(getattr(patient, attr)))
    return result

def generate_patient_summary(patient_id: str, report_template: str) -> str:
    # Log the request
    logger.info(f"Generating patient summary - Patient ID: {patient_id}")
    logger.info(f"Report template: {report_template}")
    
    # Define allowed and restricted attributes
    allowed_attributes = {'name', 'age', 'gender', 'id'}
    restricted_attributes = {'diagnosis', 'treatment_plan'}
    
    try:
        # Pipeline: retrieve -> validate -> fill
        patient = retrieve_patient(patient_id, patients_data)
        validate_template_access(report_template, restricted_attributes)
        result = fill_template(report_template, patient, allowed_attributes)
        
        # Log successful generation
        logger.info(f"Successfully generated summary for patient {patient_id}")
        logger.info(f"Generated summary: {result}")
        
        return result
    except ValueError as e:
        # Log errors
        logger.error(f"Failed to generate summary for patient {patient_id}: {str(e)}")
        raise
