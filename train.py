import torch
import torch.optim as optim
from model import MiniLLM
from utils import tokenize
from config import config

hp = config['hyperparameters']
train_cfg = config['training']
vocab = hp['vocab']
vocab_size = len(vocab)

model = MiniLLM(vocab_size, hp['embedding_dim'], hp['hidden_dim'], hp['num_layers'])
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])

tokenized_data = [tokenize(sentence, vocab) for sentence in train_cfg['data']]

for epoch in range(hp['num_epochs']):
    for sentence in tokenized_data:
        for i in range(1, len(sentence)):
            input_seq = torch.tensor(sentence[:i]).unsqueeze(0)
            target = torch.tensor(sentence[i]).unsqueeze(0)
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output[:, -1, :], target)
            loss.backward()
            optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "mini_llm.pth")