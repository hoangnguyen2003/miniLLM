import torch
from model import MiniLLM
from utils import tokenize
from config import config

hp = config['hyperparameters']
vocab = hp['vocab']
vocab_size = len(vocab)

model = MiniLLM(vocab_size, hp['embedding_dim'], hp['hidden_dim'], hp['num_layers'])
model.load_state_dict(torch.load("mini_llm.pth"))

def predict(input_text):
    input_tokens = tokenize(input_text, vocab)
    input_tensor = torch.tensor(input_tokens).unsqueeze(0)
    output = model(input_tensor)
    predicted_token = torch.argmax(output[:, -1, :]).item()
    return list(vocab.keys())[list(vocab.values()).index(predicted_token)]

if __name__ == "__main__":
    input_text = "hello world how"
    prediction = predict(input_text)
    print(f"Input: {input_text}, Predicted: {prediction}")