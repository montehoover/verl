from typing import AnyStr

def process_input_text(txt: AnyStr) -> str:
    """
    Process input text by converting it to lowercase for standardization.
    
    Args:
        txt: Input text to process
        
    Returns:
        Lowercased version of the input text
    """
    if isinstance(txt, bytes):
        txt = txt.decode('utf-8')
    return txt.lower()
