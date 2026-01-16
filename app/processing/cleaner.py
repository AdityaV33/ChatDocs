def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    """
    if not text:
        return ""

    
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text
