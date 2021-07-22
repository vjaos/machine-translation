import torch
import torch.nn as nn
import random
import torch.optim as optim
import gc

from train_utils import prepare_dataset, prepare_iterators, train, evaluate, SRC_FIELD, TRG_FIELD

gc.collect()
torch.cuda.empty_cache()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        encoder_states, hidden = self.rnn(embedded)

        return encoder_states, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, hidden, encoder_states):
        src = src.unsqueeze(0)
        embedded = self.dropout(self.embedding(src))
        decoder_state, hidden = self.rnn(embedded, hidden)
        prediction = self.out(decoder_state.squeeze(0))

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ration=0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_states, hidden = self.encoder(src)

        decoder_input = trg[0, :]  # <SOS> token

        for i in range(1, max_len):
            decoder_prediction, hidden = self.decoder(decoder_input, hidden, encoder_states)

            outputs[i] = decoder_prediction
            top1 = decoder_prediction.max(1)[1]

            use_teacher_forcing = random.random() < teacher_forcing_ration
            decoder_input = trg[i] if use_teacher_forcing else top1

        return outputs


if __name__ == '__main__':
    import time
    import datetime

    batch_dim = 8
    train_data, valid_data, test_data = prepare_dataset('../datasets/eng-de.tsv')
    train_iter, valid_iter, test_iter = prepare_iterators(train_data, valid_data, test_data, batch_dim, DEVICE)

    n_epochs = 50
    clip_bound = 1
    input_dimension = len(SRC_FIELD.vocab)
    output_dimension = len(TRG_FIELD.vocab)
    n_layers = 2
    embedding_dim = 512
    hidden_dimension = 1024
    dropout_ration = 0.5

    model = Seq2Seq(
        encoder=Encoder(input_dimension, embedding_dim, hidden_dimension, n_layers, dropout_ration).to(DEVICE),
        decoder=Decoder(output_dimension, embedding_dim, hidden_dimension, n_layers, dropout_ration).to(DEVICE),
        device=DEVICE
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters())

    pad_idx = TRG_FIELD.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        start = time.time()

        train_loss = train(model, train_iter, optimizer, criterion, clip_bound)
        valid_loss = evaluate(model, valid_iter, criterion)

        end = time.time()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')

        diff = end - start
        remaining_secs = (n_epochs - epoch + 1) * diff
        print(
            f"Epoch: {epoch + 1} "
            f"| Epoch time: {datetime.timedelta(seconds=diff)} "
            f"| Remaining time: {datetime.timedelta(seconds=remaining_secs)}")
        print(f"\tTrain loss: {train_loss:.3f}")
        print(f"\tValid loss: {valid_loss:.3f}")
        print("----------------------------------------------------------------------")
