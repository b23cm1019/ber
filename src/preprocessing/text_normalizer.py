import re
import unicodedata
from unidecode import unidecode

class TextNormalizer:

    @staticmethod
    def normalize(text):

        if not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFKC", text)
        text = unidecode(text)
        text = text.lower()

        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()