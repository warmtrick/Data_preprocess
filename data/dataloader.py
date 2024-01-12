import torch
import json
import os
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from.tokenizer import jieba_tokenizer

def build_vocab_from_iterator(iterator, specials=('<pad>', '<unk>'), min_freq=1):
    vocab = {word: idx for idx, word in enumerate(specials)}
    word_counts = {}

    for text in iterator:
        for word in text:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    for word, count in word_counts.items():
        if count >= min_freq and word not in vocab:
            idx = len(vocab)
            vocab[word] = idx
    
    return vocab

def load_data_and_build_vocab(data_path, max_length, tokenizer, vocab_path):
    texts, labels = [],[]
    label_map = {}
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            data = json.loads(line)
            text, label = data['text'].strip(), data['label'].strip()

            if label not in label_map:
                label_map[label] = len(label_map)
            
            texts.append(text[:2000]) # 限制文本长度，注意和token长度区分
            labels.append(label_map[label])
    
    if os.path.exists(vocab_path):
        vocab = torch.load(vocab_path)
    else:
        vocab_generator = (tokenizer(text) for text in tqdm(texts))
        vocab = build_vocab_from_iterator(vocab_generator)
        torch.save(vocab, vocab_path)

    num_classes = len(label_map)
    return texts, labels, vocab, label_map, num_classes

def text_to_tensor(text, vocab, max_length):
    tokens = [vocab.get(token, vocab['<unk>']) for token in jieba_tokenizer(text)]
    if len(tokens) < max_length:
        tokens.extend([vocab['<pad>']] * (max_length - len(tokens)))
    else:
        tokens = tokens[:max_length]
    return torch.tensor(tokens, dtype=torch.long)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = text_to_tensor(self.texts[idx], self.vocab, self.max_length)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return text, label

def split_and_instantiate_dataset(texts, labels, vocab, max_length, train_ratio=0.8, 
        val_ratio=0.1, test_ratio=0.1, random_seed=42):
    random.seed(random_seed)

    total_samples = len(texts)
    indices = list(range(total_samples))
    random.shuffle(indices)

    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:(train_size + val_size)]
    test_indices = indices[(train_size + val_size):]

    train_dataset = TextClassificationDataset(
        [texts[idx] for idx in train_indices],
        [labels[idx] for idx in train_indices],
        vocab, max_length)
    
    val_dataset = TextClassificationDataset(
        [texts[idx] for idx in val_indices],
        [labels[idx] for idx in val_indices],
        vocab, max_length)
    
    test_dataset = TextClassificationDataset(
        [texts[idx] for idx in test_indices],
        [labels[idx] for idx in test_indices],
        vocab, max_length)

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(config):
    tokenizer = jieba_tokenizer
    texts, labels, vocab, label_map, num_classes = load_data_and_build_vocab(config.get('dataset_path'), config.get('max_length'), tokenizer, config.get('vocab_path'))

    train_dataset, val_dataset, test_dataset = split_and_instantiate_dataset(texts, labels, vocab, config.get('max_length'))

    train_dataloader = DataLoader(train_dataset, batch_size=config.get('batch_size'), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.get('batch_size'), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.get('batch_size'), shuffle=True)

    vocab_size = len(vocab)

    return train_dataloader, val_dataloader, test_dataloader, label_map, vocab_size, num_classes
