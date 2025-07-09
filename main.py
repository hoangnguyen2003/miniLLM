import torch
import torch.optim as optim
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import MiniLLM
from utils import tokenize
from config import config
from tqdm.auto import tqdm
import os, argparse

def main(args):
    with open(config['dataset_path'], 'r', encoding='utf-8') as file:
        text = file.read()

    tokenizer, max_sequence_len, input_sequences = tokenize(text)
    model = MiniLLM(
        len(tokenizer.word_index) + 1,
        config['hyperparameters']['embedding_dim'],
        config['hyperparameters']['hidden_dim'],
        config['hyperparameters']['num_layers']
    )
             
    if args.mode == 'train':
        X = input_sequences[:, :-1]
        y = input_sequences[:, -1]

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['hyperparameters']['learning_rate'])

        for epoch in range(config['hyperparameters']['epochs']):
            for i in tqdm(range(len(y)), desc=f'Epoch {epoch + 1}', unit='sample'):
                input_seq = torch.tensor(X[i]).unsqueeze(0)
                target = torch.tensor(y[i]).unsqueeze(0).type(torch.LongTensor)
                optimizer.zero_grad()
                output = model(input_seq)
                loss = criterion(output[:, -1, :], target)
                loss.backward()
                optimizer.step()
            print(f'loss: {loss.item()}')

        os.makedirs(os.path.dirname(config['model_path']), exist_ok=True)
        torch.save(model.state_dict(), config['model_path'])
        print(f'Model saved to {config['model_path']}')

    else:
        model.load_state_dict(torch.load(config['model_path']))

        res = args.input
        for _ in range(args.num_predictions):
            input_tokens = tokenizer.texts_to_sequences([res])[0]
            input_tokens = pad_sequences([input_tokens], maxlen=max_sequence_len - 1, padding='pre')
            input_tensor = torch.tensor(input_tokens[0]).unsqueeze(0)
            predicted_token = torch.argmax(model(input_tensor)[:, -1, :]).item()
            res += ' ' + tokenizer.index_word.get(predicted_token, '')

        print(f'Predict: {res}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='miniLLM'
    )
    parser.add_argument(
        '--mode',
        required=True,
        choices=['train', 'predict'],
        help='Mode to run: "train" or "predict"'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Text input for predict mode'
    )
    parser.add_argument(
        '--num_predictions',
        type=int,
        help='Number of words to predict',
        default=1
    )

    main(parser.parse_args())