import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from data.dataloader import create_dataloaders, load_data_and_build_vocab, split_and_instantiate_dataset
from models.model import TextCNN
from utils.config import Config
from evaluate import evaluate_model

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_classes, num_epochs, device, result_save_path):
    print('Training started')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}')

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            print("模型输出：", outputs)
            print("模型输出形状：", outputs.shape)
            print("标签", labels)

            if num_classes == 2:
                # loss = criterion(outputs, labels.view(-1, 1).float())
                print(outputs.squeeze())
                print(labels.float())
                loss = criterion(outputs.squeeze(), labels.float())  # 去除单维度
            else:
                loss = criterion(outputs, labels.long())  # CrossEntropyLoss 需要 long 类型的标签

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step = epoch * len(train_dataloader) + batch_idx
            progress_bar.set_postfix({'Loss': f'{total_loss / (batch_idx + 1):.4f}'})

        avg_loss, report, conf_matrix, accuracy = evaluate_model(model, val_dataloader, criterion, num_classes, device, result_save_path)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}')

    print('Training completed')

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = Config.get_config()
    config.replace_project_name_placeholder() # 替换配置文件中的占位符
  
    train_dataloader, val_dataloader, test_dataloader, label_map, vocab_size, num_classes = create_dataloaders(config)

    model = TextCNN(vocab_size, config.get('embed_dim'), config.get('num_filters'), config.get('filter_sizes'), config.get('max_length'), num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.get("learning_rate"))

    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_classes, config.get("num_epochs"), device, config.get("result_save_path"))