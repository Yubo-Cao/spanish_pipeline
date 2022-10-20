import unicodedata
from pathlib import Path

from docx import Document
from yaml import Dumper, dump

doc_path = Path("vocab.docx")
doc = Document(doc_path)
tables = doc.tables

vocab_list = []
for table in tables:
    vocab_list.extend(
        [
            [l, r]
            for row in table.rows
            if len(cells := row.cells) == 2
            and (l := cells[0].text)
            and (r := cells[1].text)
        ]
    )


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    for l, r in [("🡪", "->"), ("🡨", "<-"), ("🡫", "^"), ("🡬", "v"), ('‘', "'"), ('’', "'")]:
        text = text.replace(l, r)
    return text.strip()

vocab_list = [[normalize(l), normalize(r)] for l, r in vocab_list]

with open("cards.yml", "w", encoding='utf-8') as f:
    dump(vocab_list, f, Dumper=Dumper, encoding="utf-8", allow_unicode=True)
