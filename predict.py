# predict.py
import torch
from models.model import TextCNN
from data.dataloader import load_data_and_build_vocab
from data.tokenizer import jieba_tokenizer
from utils.config import Config

def predict(text, model, vocab, max_length, device):
    model.eval()
    tokens = [vocab[token] for token in jieba_tokenizer(text)]
    if len(tokens) < max_length:
        tokens.extend([vocab['<pad>']] * (max_length - len(tokens)))
    else:
        tokens = tokens[:max_length]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.round(output).item()
    return prediction

if __name__ == "__main__":
    config = Config('config.json')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, _, vocab = load_data_and_build_vocab(config.dataset_path, jieba_tokenizer, config.max_length, config.vocab_path)
    model = TextCNN(len(vocab), config.embed_dim, config.num_filters, config.filter_sizes, config.max_length, config.num_classes).to(device)
    model.load_state_dict(torch.load(config.model_path))
    model.to(device)
    
    # 示例文本预测
    sample_text = "这里填入待预测的文本"
    print(f"Predicted class for the sample text: {predict(sample_text, model, vocab, config.max_length, device)}")
