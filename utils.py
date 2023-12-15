import json
from typing import Iterable
from langchain.schema import Document
from langchain.document_loaders import TextLoader

# Code is adapted from https://github.com/langchain-ai/langchain/issues/3016
def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

if __name__ == "__main__":
    # Simple load and save test.
    loader = TextLoader("./test/OR_README.md")
    final = loader.load()
    save_docs_to_jsonl(final,'./tempdata/data.jsonl')
    final2=load_docs_from_jsonl('./tempdata/data.jsonl')
    assert len(final) == len(final2)