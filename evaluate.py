# evaluate.py
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_model(model, data_loader, criterion, num_classes, device, result_save_path):
    model.eval()
    corrects, total_loss = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if num_classes == 2:
                # loss = criterion(outputs, labels.view(-1, 1).float())
                loss = criterion(outputs.squeeze(), labels.float())  # 去除单维度
            else:
                loss = criterion(outputs, labels.long())  # CrossEntropyLoss 需要 long 类型的标签
            total_loss += loss.item()

            if num_classes == 2:
                predicted_labels = torch.sigmoid(outputs.squeeze()) 
                print("sigmoid后的输出：", predicted_labels)
                # predicted_labels = torch.round(outputs).view(-1)
                predicted_labels = (predicted_labels >= 0.5).float()
                print("阈值后的输出：", predicted_labels)
            else:
                predicted_labels = torch.argmax(outputs, dim=1)  # 多分类，选择概率最高的类别

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())
            corrects += (predicted_labels == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    print("每个dataloader总损失：", avg_loss)
    avg_loss = total_loss / len(data_loader.dataset)
    print("平均损失：", avg_loss)
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)

    with open(result_save_path, 'w') as file:
        file.write("Classification Report:\n")
        file.write(str(report))
        file.write("\n\nConfusion Matrix:\n")
        file.write(str(conf_matrix))
        file.write("\n\nAccuracy:\n")
        file.write(str(accuracy))
    
    return avg_loss, report, conf_matrix, accuracy