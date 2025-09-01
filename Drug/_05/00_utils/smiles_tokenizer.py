# 00_utils/smiles_tokenizer.py
import torch
from transformers import AutoTokenizer

def get_tokenizer(model_name='seyonec/ChemBERTa-zinc-base-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_smiles(smiles_list, tokenizer, max_len=128):
    return tokenizer(smiles_list,
                     return_tensors='pt',
                     padding='max_length',
                     truncation=True,
                     max_length=max_len)

# 00_utils/smiles_tokenizer.py (이미 있다면 이어서 추가)

def smiles_char_tokenizer(smiles_list):
    all_chars = sorted(list(set("".join(smiles_list))))
    stoi = {ch: i + 1 for i, ch in enumerate(all_chars)}  # padding 0
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi) + 1
    return stoi, itos, vocab_size

from collections import defaultdict

def build_vocab(smiles_list):
    chars = set()
    for smiles in smiles_list:
        chars.update(smiles)
    chars = sorted(list(chars))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}  # 0은 padding 용도
    stoi['<pad>'] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode_smiles(smiles_list, stoi, max_len):
    encoded = []
    for smiles in smiles_list:
        ids = [stoi.get(ch, 0) for ch in smiles]  # 없는 char는 0으로
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        encoded.append(ids)
    return encoded
