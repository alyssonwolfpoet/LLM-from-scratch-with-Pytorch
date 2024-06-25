# Importar bibliotecas necessárias
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import cross_entropy

# Carregar dados de treinamento
train_data = pd.read_csv('train.csv')

# Criar tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Criar classe de dataset personalizada
class BertDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Criar datasets de treinamento e validação
train_dataset = BertDataset(train_data, tokenizer)
val_dataset = BertDataset(val_data, tokenizer)

# Criar data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Criar modelo BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)

# Definir função de perda e otimizador
criterion = cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Treinar modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predicted = torch.max(outputs.scores, 1)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / len(val_dataset)
    print(f'Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}')

# Salvar modelo treinado
torch.save(model.state_dict(), 'bert_llm.pth')