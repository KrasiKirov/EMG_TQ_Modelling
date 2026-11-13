"""Pronunciation pass: narration text (with symbols and digits) -> text Kokoro reads correctly.

Rules (see STORYBOARD.md §2): letters for codes and acronyms, units spelled out, numbers as words
(digits after the decimal point one by one), minus / plus signs spoken. Kokoro's front end (misaki)
accepts [text](/phonemes/) for words it would otherwise mispronounce: "T A" becomes "tee uh"
without it, so TA gets explicit phonemes.
"""
import re

from num2words import num2words

DIGITS = 'zero one two three four five six seven eight nine'.split()
WORDS = [  # whole-word replacements, applied in this order (case-sensitive)
    ('R²', 'R squared'), ('Nm²', 'newton metres squared'), ('Nm', 'newton metres'), ('mrad', 'milliradians'),
    ('kHz', 'kilohertz'), ('Hz', 'hertz'), ('EMG', 'E M G'), ('LSTM', 'L S T M'), ('MVC', 'M V C'),
    ('D01', 'D zero one'), ('C01', 'C zero one'), ('Nadam', 'N-Adam'), ('NumPy', 'num pie'),
    ('GM', 'G M'), ('GL', 'G L'), ('SOL', 'soleus'), ('TA', '[T A](/tˈi ˈA/)'),
    ('CH', 'C H'), ('EG', 'E G'), ('HM', 'H M'), ('IES', 'I E S'), ('JM', 'J M'), ('YES', 'Y E S'),
]
UNIT_AFTER_NUMBER = [('rad', 'radians'), ('s', 'seconds'), ('ms', 'milliseconds')]
SINGULAR = [(re.compile(r'(?<![\d.,])\b(1|[Oo]ne) Nm\b'), r'\1 newton metre'), (re.compile(r'\bper Nm\b'), 'per newton metre')]


def number_words(tok):
    sign = ''
    if tok[0] in '−-':
        sign, tok = 'minus ', tok[1:]
    elif tok[0] == '+':
        sign, tok = 'plus ', tok[1:]
    tok = tok.replace(',', '')
    if '.' in tok:
        whole, frac = tok.split('.')
        return sign + num2words(int(whole or 0)) + ' point ' + ' '.join(DIGITS[int(c)] for c in frac)
    return sign + num2words(int(tok))


NUM = re.compile(r'(?<![\w.])([−+-]?\d[\d,]*(?:\.\d+)?)(?![\w])')


def speak(text):
    for a, b in UNIT_AFTER_NUMBER:                            # "0.2 rad" -> "0.2 radians"
        text = re.sub(rf'(\d)\s*{a}\b', rf'\1 {b}', text)
    for pat, rep in SINGULAR:                                 # "1 Nm", "per Nm" -> "newton metre"
        text = pat.sub(rep, text)
    for a, b in WORDS:
        text = re.sub(rf'(?<![\w\[/]){re.escape(a)}(?![\w\]])', b, text)
    text = text.replace('%', ' percent').replace('×', ' times ').replace('±', 'plus or minus ')
    text = NUM.sub(lambda m: number_words(m.group(1)), text)
    return re.sub(r'\s+', ' ', text).strip()


if __name__ == '__main__':
    for t in ['EG at −0.038 rad, R² 0.986, MAE 0.84 Nm, 56,896 windows, 30 % dropout, +0.25 rad, 1000 samples',
              'GM, GL, SOL and TA; D01 and C01; 2 Hz; 1 kHz; 0.0001; 5 mrad; 14,372 Nm².',
              'One Nm of error; at most 1 Nm; size per Nm of push; 0.1 Nm; 21 Nm.']:
        print(speak(t))
