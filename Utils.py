import re

special_char_mapping = {
    "†": "+", "∞": "infinity", "×": "*", "°": "degree", "–": "-", "→": "reaction", "~": "negation", "・": "*", "—": "-",
    "€": "belongs to", "√": "sqrt", "•": "*", "」": "floor", "ст": "ct",
}

mis_spelled = {
    "tany": "tan y", "thea": "theta", "bycm": "by centimetre", "wihcha": "which a", "trainglea": "trianlge a",
    "liner": "linear", "whena": "when a", "thenn": "then n", "diag": "diagonal", "inx": "in x", "acos": "a cos",
    "tano": "tan o", "aarray": "an array", "andv": "and v", "ande": "and e", "givena": "given a", "thenc": "then c",
    "fory": "for y", "thatc": "that c", "withp": "with p", "ofi": "of i", "letr": "let r", "thecost": "the cost",
    "findk": "find k", "andh": "and h", "andz": "and z", "findb": "find b", "asa": "as a", "letp": "let p",
    "ifc": "if c", "wherei": "where i", "areacm": "area cm", "findm": "find m", "ife": "if e", "ifz": "if z",
    "onr": "on r", "asf": "as f", "isy": "is y", "isp": "is p", "bea": "be a", "equationa": "equation a",
    "ands": "and s", "byy": "by y", "liney": "line y", "ofc": "of c", "atp": "at p", "ofn": "of n", "thatcm": "that cm",
    "tocm": "to cm", "theny": "then y", "ifr": "if r", "parabolay": "parabola y", "ifb": "if b", "sas": "s as",
    "ofcosec": "of cosec", "equationy": "equation y", "thenk": "then k", "sidescm": "sides cm", "sol": "solution",
    "tob": "to b", "cotx": "cot x", "ina": "in a", "ing": "in g", "froma": "from a", "fora": "for a", "byr": "by r",
    "ofk": "of k", "sina": "sin a", "ando": "and o", "thatb": "that b", "thatx": "that x", "arrayarray": "array",
    "arraya": "array a", "ifd": "if d", "pointsp": "points p", "planex": "plane x", "isx": "is x",
    "vectori": "vector i", "thenb": "then b", "incm": "in cm", "ifi": "if i", "cosa": "cos a", "cose": "cos e",
    "cosx": "cos x", "sino": "sin o", "whenx": "when x", "ifm": "if m", "relationr": "relation r",
    "vectorsi": "vectors i", "linesx": "lines x", "thatarray": "that array", "tox": "to x", "findp": "find p",
    "ofb": "of b", "andarray": "and array", "sidecm": "side cm", "betweena": "between a", "heightcm": "height cm",
    "lengthcm": "length cm", "ofy": "of y", "bya": "by a", "ofarray": "of array", "findf": "find f", "andq": "and q",
    "ifcm": "if cm", "findx": "find x", "matrixa": "matrix a", "pointp": "point p",
    "monthlyarray": "monthly array", "carray": "c array", "plancml": "plan cm l", "adjecnt": "adjacent",
    "floorm": "floor m", "rectanglein": "rectangle in", "whichn": "which n", "thann": "than n", "simpify": "simplify",
    "productba": "product ba", "rotat": "rotate", "whenu": "when", "thatqy": "that qy", "ifmm": "if mm",
    "papercm": "paper cm", "neeed": "need", "emaining": "remaining", "ammmm": "am", "and cst": "and cst",
    "ploynomars": "polynomers", "ifxx": "if xx"
}

keep_spec_char = '+-=<>%*/^'


def filter_correct(sentence):
    """
    Correct the spellings and change special characters names
    """
    splitted = sentence.split(' ')
    dummy_list = []
    for token in splitted:
        if token.isalnum() and token.isascii() and token != '_':  # if token is alpha numeric and within ASCII range
            if token in mis_spelled:
                dummy_list.append(mis_spelled[token])
            else:
                dummy_list.append(token)
        else:  # if it is special char
            if token in keep_spec_char:
                dummy_list.append(token)
            elif token in special_char_mapping:
                dummy_list.append(special_char_mapping[token])

    return ' '.join(dummy_list)


def remove_single_word_num(sent):
    """
    Remove numbers and words of single length such as "x + 23 y - abc" will become "+ - abc"
    """
    dummy_list = []

    for token in sent.split():
        if (not token.isdigit()) and ((token.isalpha() and len(token) > 1) or (not token.isalnum())):
            dummy_list.append(token)

    return ' '.join(dummy_list)


def insert_spaces(sentence):
    """
    Add a space around special characters, number and digits. So "2x+y -1/3x" becomes: "2 x + y - 1 / 3 x"
    """
    dummy_list = []
    splitted_sent = list(sentence)

    for i in range(len(splitted_sent) - 1):
        dummy_list.append(splitted_sent[i])

        if splitted_sent[i].isalpha():  # if it is an alphabet
            if splitted_sent[i + 1].isdigit() or (not splitted_sent[i + 1].isalnum()):
                dummy_list.append(' ')

        elif splitted_sent[i].isdigit():  # if it is a number
            if splitted_sent[i + 1].isalpha() or (not splitted_sent[i + 1].isalnum()):
                dummy_list.append(' ')

        elif (not splitted_sent[i].isalnum()) and (
                splitted_sent[i] not in [' ', '\\']):  # if it is a special char but not ' ' already
            if splitted_sent[i + 1].isalnum():
                dummy_list.append(' ')

    dummy_list.append(splitted_sent[-1])

    return ''.join(dummy_list)


def preprocess(a):
    # convert the characters into lower case
    a = a.lower()

    # remomve newline character
    a = re.sub("\\n", " ", a)

    # remove the pattern [ whatever here ]. Use { } or  ( ) in place of [ ] in regex
    a = re.sub(r"\[(.*?)\]", ' ', a)

    # remove abbrevationns like I.I.T. , JEE, J.e.e. etc
    a = re.sub(r"(?<!\S)(?:(?:[A-Za-z]\.){3,}|[A-Z]{3,})(?!\S)", ' ', a)

    # remove Questions beginners Q5. 5. question 5.
    a = re.sub(r"^[\w]+(\s|\.)(\s|\d+(\.*(\d+|\s)))\s*", " ", a)

    # remove MathPix markdown starting from \( and ending at \) while preserving data inside \text { preserve this }
    a = re.sub(r'\s*\\+\((.*?)\\+\)', lambda x: " ".join(re.findall(r'\\[a-z]{3,}\s*{([^{}]*)}', x.group(1))), repr(a))

    # remove options from questions i.e character bounded by () given there is no spacing inside ()
    a = re.sub(r"\s*\([^)\s]*\)\s*", " ", a)

    # remove data inside {} -> at max 2 characters {q.}, {5.}
    a = re.sub(r"{.{0,2}}", " ", a)

    # Insert spaces among spec chars, digits and nums
    a = insert_spaces(a)

    # remove whatever comes after \\ double or single slashes except space
    a = re.sub(r"(\\[^ ]+)", ' ', a)

    # remove strings which are not ASCII, correct spellings, map special characters
    a = filter_correct(a)

    # remomve newline character inserted by raw  string in the removal of Mathpix
    a = re.sub("\\n", " ", a)

    # remove any repeating special character (more than one times) except space.  So it'll remove ._  ++ etc +-=
    # except spaces
    a = re.sub(r"([^a-zA-Z0-9 ]{2,})", ' ', a)

    # remove repeated space if there is any
    a = re.sub(r"\s+", " ", a)

    return a
