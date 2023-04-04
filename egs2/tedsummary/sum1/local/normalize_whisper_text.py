from whisper.normalizers import EnglishTextNormalizer
import sys 
import re 


normalizer = EnglishTextNormalizer()
with open("data/numbers_map","r") as f:
    numbers_map = {line.strip().split(": ")[0] : line.strip().split(": ")[1] for line in f.readlines() }

# r"Mr": "mister ",
#             r"Mrs": "missus ",
#             r"St": "saint ",
#             r"Dr": "doctor ",
#             r"Prof": "professor ",
#             r"Capt": "captain ",
#             r"Gov.": "governor ",
#             r"Ald.": "alderman ",
#             r"Gen.": "general ",
#             r"Sen": "senator ",
#             r"rep": "representative ",
#             r"pres": "president ",
#             r"rev": "reverend ",
#             r"hon": "honorable ",
#             r"asst": "assistant ",
#             r"assoc": "associate ",
#             r"lt": "lieutenant ",
#             r"col": "colonel ",
#             r"jr": "junior ",
#             r"sr": "senior ",
#             r"esq": "esquire ",


replacers = {
    # common contractions
    r"\bwon't\b": "will not",
    r"\bIt'll\b": "It will",
    r"\bYou're\b": "You are",
    r"\bI'll\b": "I will",
    r"\bI'm\b": "I am",
    r"\bI've\b": "I have",
    r"\bI'd\b": "I would",
    r"\bIt's\b": "It is",
    r"\bcan't\b": "can not",
    r"\blet's\b": "let us",
    r"\bain't\b": "aint",
    r"\by'all\b": "you all",
    r"\bwanna\b": "want to",
    r"\bgotta\b": "got to",
    r"\bgonna\b": "going to",
    r"\bi'ma\b": "i am going to",
    r"\bimma\b": "i am going to",
    r"\bwoulda\b": "would have",
    r"\bcoulda\b": "could have",
    r"\bshoulda\b": "should have",
    r"\bma'am\b": "madam",
    # contractions in titles/prefixes
    r"\bmr\b": "mister ",
    r"\bmrs\b": "missus ",
    r"\bst\b": "saint ",
    r"\bdr\b": "doctor ",
    r"\bprof\b": "professor ",
    r"\bcapt\b": "captain ",
    r"\bgov\b": "governor ",
    r"\bald\b": "alderman ",
    r"\bgen\b": "general ",
    r"\bsen\b": "senator ",
    r"\brep\b": "representative ",
    r"\bpres\b": "president ",
    r"\brev\b": "reverend ",
    r"\bhon\b": "honorable ",
    r"\basst\b": "assistant ",
    r"\bassoc\b": "associate ",
    r"\blt\b": "lieutenant ",
    r"\bcol\b": "colonel ",
    r"\bjr\b": "junior ",
    r"\bsr\b": "senior ",
    r"\besq\b": "esquire ",
    # prefect tenses, ideally it should be any past participles, but it's harder..
    r"'d been\b": " had been",
    r"'s been\b": " has been",
    r"'d gone\b": " had gone",
    r"'s gone\b": " has gone",
    r"'d done\b": " had done",  # "'s done" is ambiguous
    r"'s got\b": " has got",
    # general contractions
    r"n't\b": " not",
    r"'re\b": " are",
    # r"'s\b": " is",
    r"'d\b": " would",
    r"'ll\b": " will",
    r"'t\b": " not",
    r"'ve\b": " have",
    r"'m\b": " am",

}

def contraction_normalizer(s):
    for pattern, replacement in replacers.items():
        s = re.sub(pattern, replacement, f"{s}")
    
    s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
    s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
    s = re.sub('([.,!?()])', r' \1 ', s)
    for x,y in numbers_map.items():
        s = re.sub(f" {x} ",f" {y} ",f"{s}")
    s = re.sub("(^|[.?!])\s*([a-zA-Z])", lambda p: p.group(0).upper(), s)
    s = re.sub('\s{2,}', ' ', s)
    return s


for line in sys.stdin:
    utt = line.strip().split()[0]
    text = " ".join(line.strip().split()[1:])
    #line = normalizer.standardize_numbers(text)
    line = contraction_normalizer(line.strip()).strip()
    print(f"{line}")                              
