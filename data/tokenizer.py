# data/tokenizer.py
import jieba

def jieba_tokenizer(text):
    return [word for word in jieba.cut(text) if word.strip()]
