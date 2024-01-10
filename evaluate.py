# evaluate.py
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    corrects, avg_loss = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())
            avg_loss += loss.item()
            predicted_labels = torch.round(outputs).view(-1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())
            corrects += (predicted_labels == labels).sum().item()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    avg_loss /= len(data_loader.dataset)
    return accuracy, precision, recall, f1