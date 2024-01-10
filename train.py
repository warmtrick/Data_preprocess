import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from data.dataloader import create_dataloaders, load_data_and_build_vocab, split_and_instantiate_dataset
from models.model import TextCNN
from utils.config import Config
from evaluate import evaluate_model

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}')

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step = epoch * len(train_dataloader) + batch_idx
            tb_writer.add_scalar('Loss/train', loss.item(), step)
            progress_bar.set_postfix({'Loss': f'{total_loss / (batch_idx + 1):.4f}'})

        accuracy, precision, recall, f1 = evaluate_model(model, val_dataloader, criterion, device)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}')

    print('Training completed')

if __name__ == "__main__":
    config = Config('config.json')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    texts, labels, vocab = load_data_and_build_vocab(config.dataset_path, jieba_tokenizer, config.max_length, config.vocab_path)
    train_dataset, val_dataset, test_dataset = split_and_instantiate_dataset(texts, labels, vocab, config.max_length)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(train_dataset, val_dataset, test_dataset, config.batch_size)

    model = TextCNN(len(vocab), config.embed_dim, config.num_filters, config.filter_sizes, config.max_length, config.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, config.num_epochs, device)