"""
This file replaces contractions with their expanded forms.
"""
import re
import sys

file_path = sys.argv[1]


replacement_dict = {
    "SOMEONE'S ": "SOMEONE IS ",
    "HERE'S ": "HERE IS ",
    "WHERE's ": "WHERE IS ",
    "SHE'LL ": "SHE WILL ",
    "AREN'T ": "ARE NOT ",
    "THAT'LL ": "THAT WILL ",
    "YOU'LL ": "YOU WILL ",
    "I'LL ": "I WILL ",
    "WE'LL ": "WE WILL ",
    "WE'RE ": "WE ARE ",
    "THEY'LL ": "THEY WILL ",
    "IT'LL ": "IT WILL ",
    "THEY'VE ": "THEY HAVE ",
    "I'VE ": "I HAVE ",
    "I'D ": "I WOULD ",
    "SHE'S ": "SHE IS ",
    "HE'S ": "HE IS ",
    "WHAT'S ": "WHAT IS ",
    "THAT'S ": "THAT IS ",
    "ISN'T ": "IS NOT ",
    "WEREN'T ": "WERE NOT ",
    "YOU'D ": "YOU WOULD ",
    "SHOULDN'T ": "SHOULD NOT ",
    "HOW'S ": "HOW IS ",
    "WASN'T ": "WAS NOT ",
    "HASN'T ": "HAS NOT ",
    "DOESN'T ": "DOES NOT ",
    "DIDN'T ": "DID NOT ",
    "DON'T ": "DO NOT ",
    "IT'S ": "IT IS ",
    "COULDN'T ": "COULD NOT ",
    "WON'T ": "WILL NOT ",
    "WOULDN'T ": "WOULD NOT ",
    "WE'VE ": "WE HAVE ",
    "HAVEN'T ": "HAVE NOT ",
    "CAN'T ": "CAN NOT ",
    "LET'S ": "LET US ",
    "THEY'RE ": "THEY ARE ",
    "YOU'RE ": "YOU ARE ",
    "YOU'VE ": "YOU HAVE ",
    "I'M ": "I AM ",
    "THERE'S ": "THERE IS ",
}


with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()
    for k, v in replacement_dict.items():
        text = re.sub(k, v, text)
    print(text)
