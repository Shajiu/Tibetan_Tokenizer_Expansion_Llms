## 词汇表扩充工具包


### 1 介绍

由于原版LLaMA对藏文的支持非常有限，本项目对Llama 2 进行藏文词表扩充，词表由32000 扩展至56724，提高模型在藏文的编解码效率。在TibetanGeneralCorpus 上使用Sentencepiece 工具训练基于Unigram 策略的藏文分词器。生成的词表与原版Llama 2 的32K 词表进行合并，排除重复的词元后，得到扩充后词表规模为56724。



### 2 数据预处理
``` python
file_dir = 'data/corpus'
file_list = [os.path.join(file_dir, path) for path in os.listdir(file_dir)]
for path in tqdm(os.listdir(file_dir)):
    with open(os.path.join(file_dir, path), 'r') as f:
        data = f.readlines()
        data = [' '.join(json.loads(d)['paras']) for d in data]

    path = path[:-6] + '.txt'
    with open(os.path.join('corpus_models', path), 'w') as f:
        f.write('\n'.join(data))
 ```
>corpus下为多个待转换的文件。每行如下：
 {"paras": "文本序列内容"}


 ### 3 训练sentencepiece
```python
start_time = time.time()
spm.SentencePieceTrainer.train(
    input='token.txt',                           # 输入训练文件路径
    model_prefix='./models/Tibetan_model',       # 模型存储路径以及前缀
    shuffle_input_sentence=False,                # 是否打乱句子
    train_extremely_large_corpus=True,           
    max_sentence_length=16384,                   # 句子最大长度
    pad_id=3,
    model_type="unigram",                        # 模型类型 bpe、char、word、unigram(gram)
    vocab_size=25000,
    split_digits=True,
    split_by_unicode_script=True,
    byte_fallback=True,
    allow_whitespace_only_pieces=True,
    character_coverage=0.9995,
    remove_extra_whitespaces=False,
    normalization_rule_name="nfkc",
)

end_time = time.time()
print(end_time - start_time)
```
>输入为：上一步生成的数据集，每行内容为：经过简单分词后的序列文本
输出为：在model_prefix下分别生成两个model和vocabs文件

### 4 合并词汇表
```python
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
```

