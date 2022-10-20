import re

import pyperclip
from pipelines import register

@register('quizlet')
def quizlet_pipeline(cards, env):
    unicode_re_escape = {
        r"\s*\.\.\.\s*": "…",
        r"\s*---\s*": "—",
        r"\s*--\s*": "–",
        r"\s*-\s*": "-",
        r"\s*/\s*": " / ",
    }

    for card in cards:
        for k, v in unicode_re_escape.items():
            card[0] = re.sub(k, v, card[0])
            card[1] = re.sub(k, v, card[1])

    quizlet = env.get_template("quizlet.jinja")
    pyperclip.copy(quizlet.render(cards=cards))
    print("Copied to clipboard!")
