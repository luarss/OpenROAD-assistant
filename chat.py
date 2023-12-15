from llama_cpp import Llama
from langchain.prompts import ChatPromptTemplate
import argparse
from utils import load_docs_from_jsonl
from embed import load_existing_embeddings, generate_new_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--gen-embeddings", action="store_true", help="Regenerate embeddings")
parser.add_argument("--max-tokens", type=int, help="max_tokens hyperparameter", default = 200)
parser.add_argument("--temp", type=float, help="temperature hyperparameter", default = 0.2)
parser.add_argument("--n-ctx", type=int, help="n_ctx hyperparameter", default = 8000)
args = parser.parse_args()
gen_embed = args.gen_embeddings
max_tokens = args.max_tokens
temp = args.temp
n_ctx = args.n_ctx

def create_completion(db, question,
                        max_tokens, temperature, n_ctx):
    # Define the prompt template
    template = PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_template(template)
    
    # Step 6: Generate a query and search for relevant chunks
    context = "\n---\n".join(x[0].page_content for x in db.similarity_search_with_score(question, k = 8))
    final_prompt = prompt.format_messages(context = context, question = question)[0].content
    print(final_prompt)
    
    # Step 7: Use llama-cpp-python as a prototype. 
    llm = Llama(model_path="../llama.cpp/models/mistral-instruct/ggml-model-q4_0.gguf", n_ctx = n_ctx)
    output = llm.create_completion(final_prompt,
                                   suffix=None,
                                   max_tokens=max_tokens, # set this to 0 for no limit on tokens (depend on n_ctx)
                                   temperature=temperature, # higher temperature, less factual.
                                   top_p=0.95,
                                   logprobs=None,
                                   echo=False,
                                   stop=[],
                                   frequency_penalty=0.0,
                                   presence_penalty=0.0,
                                   repeat_penalty=1.1,
                                   top_k=40,
                                   stream=False,
                                   tfs_z=1.0,
                                   mirostat_mode=0,
                                   mirostat_tau=5.0,
                                   mirostat_eta=0.1,
                                   model=None,
                                   stopping_criteria=None,
                                   logits_processor=None)
    return context, output

MISTRAL_START= "<s>[INST] "
MISTRAL_END=" [/INST]"
TRIPLE_QUOTES = """\n\"\"\"\n"""
PRE_PROMPT_QUERY = "Pay attention and remember the information below, which will help to answer the question or imperative after the context ends.\n"
PROMPT_QUERY = "\nAccording to only the information in the document sources provided within the context above, "

PROMPT_TEMPLATE = """%s%s{context}%s%s{question}"""%(TRIPLE_QUOTES, PRE_PROMPT_QUERY, TRIPLE_QUOTES, PROMPT_QUERY)
if gen_embed: 
    chunks = load_docs_from_jsonl("./tempdata/data.jsonl")
    db, retriever = generate_new_embeddings(chunks)
else:
    db, retriever = load_existing_embeddings()
question = "What function does the OpenROAD module name odb serve?"
context, output = create_completion(db, question, max_tokens, temp, n_ctx)
print(context)
print(output["choices"][0]["text"])

