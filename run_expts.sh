#!/bin/sh

## Context chunking expt ##
python chunk.py --chunk-size 100 --overlap 20
python chat.py --gen-embeddings --filename chunk100

python chunk.py --chunk-size 150 --overlap 20
python chat.py --gen-embeddings --filename chunk150

python chunk.py --chunk-size 200 --overlap 20
python chat.py --gen-embeddings --filename chunk200

python chunk.py --chunk-size 250 --overlap 20
python chat.py --gen-embeddings --filename chunk250

python chunk.py --chunk-size 300 --overlap 20
python chat.py --gen-embeddings --filename chunk300

## Chat Hyperparams expt ##
python chunk.py --chunk-size 100 --overlap 20
python chat.py --gen-embeddings --filename ctx_0 --n-ctx 0
python chat.py --gen-embeddings --filename ctx_2000 --n-ctx 2000
python chat.py --gen-embeddings --filename ctx_4000 --n-ctx 4000

python chat.py --gen-embeddings --filename maxtokens_200 --max-tokens 200
python chat.py --gen-embeddings --filename maxtokens_100 --max-tokens 100
python chat.py --gen-embeddings --filename maxtokens_500 --max-tokens 500

python chat.py --gen-embeddings --filename temp_0_0 --temp 0.0
python chat.py --gen-embeddings --filename temp_0_2 --temp 0.2
python chat.py --gen-embeddings --filename temp_0_5 --temp 0.5
python chat.py --gen-embeddings --filename temp_0_8 --temp 0.8
python chat.py --gen-embeddings --filename temp_1_0 --temp 1.0





