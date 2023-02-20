# Spanish Pipeline

- This project aims to provide a pipeline for Spanish vocabulary learning.

## How to use

### Dependencies

I don't know why I have so much dependencies. I think it is because I am too
lazy to write my own code. I am sorry.

- `gensim` Language model necessary for visual vocab.
- `icrawler` Image crawler for visual vocab.
- `jinja2` TeX and Quizlet template rendering.
- `nltk` tokenization and lemmatization.
- `lxml` HTML parsing.
- `pyquery` HTML parsing.
- `tqdm` progress bar.
- `requests` HTTP requests.
- `pyparsing` HTML and some natural language parsing.
- `PIL` Image processing.
- `python-docx` Word document processing.
- `numpy` Image processing.

```bash
pip install -r requirements.txt
```

### Let's use it

```bash
# name your vocab file as vocab.docx
python extract_data.py
# this will extract a file called `cards.yml` in the root directory
# then run the script. by default, it will
# - copy a quizlet text into your clipboard, each vocab is separated by `\n` and definition/translation is separated by `  `
# - generate a pdf file called `flashcards.pdf` in the `output` directory. You use have MikTeX/MacTeX/TeXLive installed to compile the tex file.
# - generate a visual dictionary in `output/visual vocab.docx`
python main.py
```

- Now you have your Quizlet, printable flashcards, and visual dictionary with you.

### Specifics

- This project expects a file named `cards.yml` in the root directory. This file
  should have the format:

  ```yaml
  - - acostarse (o→ue, bota)
  - to go to bed
  ```

  - `List<List<String>>` It should have a list of cards, each card is a list
    of two strings, the first one is the front of the card (Spanish word), and
    the second one is the back of the card (English word).
  - To make your life easier, `extract_data.py` scripts at this directory
    automatically extract data from a file named `vocab.docx` in the root
    of this project into the above file format. However, it comes with limitations:
    - The documents must consist of tables that are **inline** or simply, with
      the text and not contained in some text box.
    - Only table with 2 columns, that is, the first column is the Spanish word
      and the second column is the English word that will be extracted.
    - This program **does not** distinguish the header of the table, not
      even some example sentences or redundant information. We, however, plan
      to build an RNN model to automatically extract the header of the table in
      the future.

- To run the pipeline, simply run `python main.py` in the root directory of this
  project. This, by default, would do the following things:

  - Generate a file named `flashcards.pdf` in the `output` directory. This file
    contains a printable version of the flashcards that would suit letter-size
    paper.
    - Double-sided printing is necessary to print the flashcards.
  - Generate a text, where
    - Each line is a card
    - The front of the card (Spanish word) is separated from the back of the card
      (English word) by a tab character.
    - And it copies this into your clipboard. Open Quizlet and paste it there to
      import the cards.
  - Finally, generate a visual dictionary that consists of a sample of 18 cards
    from all the cards. The results are stored in `output/visual vocab.docx`. Where
    is takes a format of:
    - Header:
      > Nombre: Student Hora: 30 minutos\`
    - Title:
      > Diccionario Visual
    - Description:
      > Escoge 18 palabras del vocabulario de esta unidad. Escribe la palabra de
      > vocabulario y una frase completa con la palabra. Dibuja una foto que
      > representa la palabra.
    - Contents:
      A table with 3 columns, 18 rows. Every 3 rows form a group
      - The first column is the Spanish word
      - The second column is an example sentence
        - The example sentence from `spanishdict.com` is automatically scraped
          and inserted into the table. `Word2Vec` and cosine similarity are used to find the most similar definition of the Spanish word.
        - If the engine fails, then a sliding window is used to determine the
          longest subword that is in the vocabulary that can be found in the
          Spanish definition. This subword's example sentence is used instead.
          If all above failed, then the example sentence is the Spanish word
          itself.
      - The third column is a picture
        - The style of the picture can be either `icon` (not recommended), or an
          `image`. The `icon` style is scraped from `flaticon.` `com`---and
          their search engine is not smart, and usually fails to produce correct
          results. The `image` style is scraped from `google.com`, and it is
          much more reliable. However, it is not guaranteed to be correct.
        - The picture, is, however, almost guaranteed to be generated when
          `image` style is used. For `icon` style, random pictures may produce,
          but it will have something, at least. The same sliding window strategy
          described above is used to handle the error.
      - Maybe, we will use stable diffusion and AWD-LSTM/GPT-3 to generate
        example sentences and images in the future.

- Of course, you don't need to run all those pipelines above:

  ```bash
  usage: main.py [-h] [--pipeline {quizlet,tex,visual-vocab,all}] [--output OUTPUT] [--var VAR] [data]

  positional arguments:
    data                  The data file to use

  options:
    -h, --help            show this help message and exit
    --pipeline {quizlet,tex,visual-vocab,all}, -p {quizlet,tex,visual-vocab,all}
                          Pipelines to run
    --output OUTPUT, -o OUTPUT
                          The directory to output to
    --var VAR, -v VAR     Variables to pass to the pipeline
  ```

  - You can specify the data file. By default, it is `cards.yml`
  - You can specify the pipeline you want to run as well. By default, it is
    `all`. `quizlet` is for generating the text for Quizlet, `tex` is for
    generating the flashcards, and `visual-vocab` is for generating the visual
    dictionary.
  - You can specify the output directory. By default, it is `output`.
  - In `var`, specify some variables in form of `k=v`. For example, if you use
    `-v student=Michael`, then the visual vocab would have `Nombre: Michael` in
    the header.
