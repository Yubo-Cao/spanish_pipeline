import re
import shutil
import subprocess
import shutil
from pathlib import Path

import pyperclip
import yaml
import yaml
# Load the YAML file
with open("cards.yml", encoding="utf-8") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

cards = [card.copy() for card in data]
# Escape and formatting a little
latex_escape_table = {
    "\\": "\\textbackslash{}",
    "{": "\\{",
    "}": "\\}",
    "$": "\\$",
    "%": "\\%",
    "&": "\\&",
    "#": "\\#",
    "_": "\\_",
    "~": "\\textasciitilde{}",
    "^": "\\^{}",
    "->": "$\\rightarrow$",
    "=>": "$\\Rightarrow$",
    "<-": "$\\leftarrow$",
    "<=": "$\\Leftarrow$",
    "<": "\\textless{}",
    ">": "\\textgreater{}",
    "€": "\\euro{}",
    "£": "\\textsterling{}",
    "§": "\\S{}",
    "°": "\\textdegree{}",
}
latex_re_escape = {
    "\s*\.\.\.\s*": r"\\ldots{}",
    "\s*---\s*": "---",
    "\s*--\s*": "--",
    r"\s*-\s*": "-",
    r"\s*/\s*": "/",
}
for card in cards:
    for k, v in latex_escape_table.items():
        card[0] = card[0].replace(k, v)
        card[1] = card[1].replace(k, v)
    for k, v in latex_re_escape.items():
        card[0] = re.sub(k, v, card[0])
        card[1] = re.sub(k, v, card[1])

# Load the Jinja2 template
env = Environment(loader=FileSystemLoader("."))
template = env.get_template("template.tex")
rendered = template.render(cards=cards)

# Render and compile the LaTeX file
temp = Path("temp.tex")
temp.write_text(rendered, encoding="utf-8")
subprocess.run(["pdflatex", "-output-directory", temp.stem, temp.name])
shutil.copyfile(temp.stem / temp.with_suffix(".pdf"), "cards.pdf")
for f in (temp.parent / temp.stem).iterdir():
    f.unlink()
(temp.parent / temp.stem).rmdir()
temp.unlink()

cards = [card.copy() for card in data]
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
