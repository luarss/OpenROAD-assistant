# Jane Austen Proof-of-concept

The goal of this is to use Retrieval Augmented Generation, or RAG, 
to generate some semblance of Austen's works by simply ingesting the 
entire written collection into vector database.

## Installation

Requires Python 3.10 and Poetry 1.6.1 ([guide](https://python-poetry.org/docs/#installing-with-the-official-installer))

Install packages to run the notebooks. There are some packages that are CPU-specific.
```
poetry install
poetry shell
```

## Todo
- Connect database to GCP Vertex AI
- Use Vertex AI to do prompt search. 
- How to evaluate models for text generation?


## Acknowledgements
The collected works are copied from Textus.io's repository
[Austen-Works](https://github.com/textvs/Austen-Works). 
The Works contained in this online electronic repository have been first 
published long before 1923; the author of the Works deceased more than 
70 years ago. It follows, world-wide, that these Works have 
entered the Public Domain and are free of copyrights. 
The texts and files contained inhere are offered for free for all to 
enjoy and to do with as one pleases.


