LEGAL_SUFFIXES = [
    "private limited",
    "pvt ltd",
    "limited",
    "ltd",
    "inc",
    "corp",
    "corporation",
    "llc",
    "llp",
    "sarl",
    "sas",
    "sa"
]

def extract_legal_form(name):

    name_lower = name.lower()

    for suffix in LEGAL_SUFFIXES:
        if name_lower.endswith(suffix):
            return suffix

    return ""

def remove_legal_form(name):

    lower = name.lower()

    for suffix in LEGAL_SUFFIXES:
        if lower.endswith(suffix):
            return lower.replace(suffix, "").strip()

    return lower