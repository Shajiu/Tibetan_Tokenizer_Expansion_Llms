# -*- coding: UTF-8 -*-

"""
功能为：合并词汇表

Authors:  shajiu
@File:  merge_tokenizers.py
Date:  2024/4/10 01:00
@Software:  PyCharm
"""
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse


def get_argparse():
    """
    基本配置
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama_tokenizer_dir', default="Llama-2-7b-hf", type=str, required=False,
                        help="The path of the original llama")
    parser.add_argument('--chinese_sp_model_file', default="/models", type=str, required=False,
                        help="New training vocabulary path")
    parser.add_argument('--model_file', default="Tibetan_model.model", type=str, required=False, help="New training vocabulary file name")

    parser.add_argument('--output_sp_dir', default="new_tokens/merged_tokenizer_sp", type=str, required=False, help="sp storage path")
    parser.add_argument('--output_hf_dir', default="new_tokens/merged_tokenizer_hf", type=str, required=False, help="Storage path in hf format")

    args = parser.parse_args()
    return args


def merge_tokenizers(args):
    """
    主要用用于合并词汇表
    :param args:
    :return:
    """
    llama_tokenizer_dir = args.llama_tokenizer_dir
    chinese_sp_model_file = args.chinese_sp_model_file
    # load
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)  # 原生LLaMA分词模型
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file+f'/{args.model_file}')

    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
    print(len(llama_tokenizer), len(chinese_sp_model))
    print(llama_tokenizer.all_special_tokens)
    print(llama_tokenizer.all_special_ids)
    print(llama_tokenizer.special_tokens_map)

    # Add Chinese tokens to LLaMA tokenizer
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
    print(len(llama_spm_tokens_set))
    print(f"Before:{len(llama_spm_tokens_set)}")
    Statistics = 0
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            Statistics += 1
            llama_spm.pieces.append(new_p)  # 将训练的分词模型追加新的token到之前的模型
    print(f"New model pieces: {len(llama_spm.pieces)}")
    print("The size of the final added vocabulary is:", Statistics)

    ## Save
    os.makedirs(args.output_sp_dir, exist_ok=True)
    with open(args.output_sp_dir + f'/{args.model_file}', 'wb') as f:
        f.write(llama_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=args.output_sp_dir + f'/{args.model_file}')

    tokenizer.save_pretrained(args.output_hf_dir)
    print(f"Chinese-LLaMA tokenizer has been saved to {args.output_hf_dir}")


def merge_tokenizers_test(args):
    """
    对合并的词汇表作测试
    :param args:
    :return:
    """
    tokenizer = LlamaTokenizer(vocab_file=args.output_sp_dir + f'/{args.model_file}')
    llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_tokenizer_dir)
    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(args.chinese_sp_model_file+ f'/{args.model_file}')
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.special_tokens_map)
    text = '''བལ་ཡུལ་བོ་དོང་དགོན་པར་ཕྱི་ཟླ་༨ཚེས་༢༣ཉིན་ས་སྐྱ་ཁྲི་འཛིན་ཞེ་གཉིས་པ་མཆོག་གིས་གང་བློ་མའི་ལྗགས་ལུང་གནང་སྐབས།'''
    print("Test text:\n", text)
    print(f"Tokenized by LLaMA tokenizer:{len(llama_tokenizer.tokenize(text))},{llama_tokenizer.tokenize(text)}")
    print(
        f"Tokenized by GoGPT-LLaMA tokenizer:{len(chinese_llama_tokenizer.tokenize(text))},{chinese_llama_tokenizer.tokenize(text)}")


if __name__ == '__main__':
    args = get_argparse()
    merge_tokenizers(args)
    merge_tokenizers_test(args)

