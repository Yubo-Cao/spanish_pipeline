import re
import shutil
import subprocess
from pathlib import Path

from pipelines import register


@register("tex")
def tex_pipeline(cards, env):
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

    jinja_env = env.get("jinja_env")
    template = jinja_env.get_template("template.tex")
    rendered = template.render(cards=cards)

    # Render and compile the LaTeX file
    temp = Path("temp.tex")
    temp.write_text(rendered, encoding="utf-8")
    subprocess.run(
        ["pdflatex", "-output-directory", temp.stem, temp.name],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    out_dir = Path(env.get("output_dir", "out"))
    shutil.copyfile(temp.stem / temp.with_suffix(".pdf"), out_dir / "flashcards.pdf")

    # Remove the temporary directory
    for f in (temp.parent / temp.stem).iterdir():
        f.unlink()
    (temp.parent / temp.stem).rmdir()
    temp.unlink()
