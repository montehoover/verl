import re
from typing import Dict, List, Union

def analyze_text(text_content: str) -> Dict[str, Union[int, List[int]]]:
    """
    Analyzes a given string to count words and find the length of each word.

    Args:
        text_content: The string to analyze.

    Returns:
        A dictionary with two keys:
        'word_count': An integer representing the total count of words.
        'word_lengths': A list containing the length of each word.
        Returns {'word_count': 0, 'word_lengths': []} if the string is empty or contains no words.
    """
    if not text_content:
        return {'word_count': 0, 'word_lengths': []}
    # Find all sequences of word characters
    words = re.findall(r'\b\w+\b', text_content)
    word_lengths = [len(word) for word in words]
    return {'word_count': len(words), 'word_lengths': word_lengths}
