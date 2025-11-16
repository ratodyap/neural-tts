import re 
from num2words import num2words

def normalize_text(text: str) -> str:

    text = text.lower().strip()
    text = re.sub(r"[\(\)\[\]{}]","",text)
    text = re.sub(r"\s+"," ",text)
    text = re.sub(r"\$","dollars",text)

    # expand numbers
    
    def replace_number(match):
        number = int(match.group())
        return num2words(number)
    
    text = re.sub(r'\d+',replace_number,text)

    return text     