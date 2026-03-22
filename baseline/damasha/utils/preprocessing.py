import re

def clean_text(text):
    """
    Removes special AI tags from the text.
    """
    return re.sub(r'</?AI_Start>|</?AI_End>', '', text) if re.search(r'</?AI_Start>|</?AI_End>', text) else text