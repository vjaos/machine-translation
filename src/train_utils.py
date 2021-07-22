import spacy
import torch
from torchtext.legacy.data import Field, BucketIterator, TabularDataset

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
ENG_TOKENIZER = spacy.load('en_core_web_sm')
DE_TOKENIZER = spacy.load('de_core_news_sm')

SRC_FIELD = Field(tokenize=lambda text: tokenize_text(text, ENG_TOKENIZER),
                  init_token=SOS_TOKEN,
                  eos_token=EOS_TOKEN,
                  lower=True)

TRG_FIELD = Field(tokenize=lambda text: tokenize_text(text, DE_TOKENIZER),
                  init_token=SOS_TOKEN,
                  eos_token=EOS_TOKEN,
                  lower=True)


def tokenize_text(text, tokenizer):
    """
    Tokenize given text by using particular tokenizer
    :param text: text in target or source language
    :param tokenizer: tokenizer for language
    :return: list of tokens
    """
    return [token.text for token in tokenizer.tokenizer(text)]


def prepare_dataset(path):
    """
    Upload dataset from give path, build vocabulary for each language and split dataset
    :param path: path to dataset
    :return: train test and valid data according to their ration
    """
    data = TabularDataset(path, format='tsv', fields=[('src', SRC_FIELD), ('trg', TRG_FIELD)])

    SRC_FIELD.build_vocab(data, min_freq=3, max_size=10_000)
    TRG_FIELD.build_vocab(data, min_freq=3, max_size=12_000)

    print(f"Source language vocab size: {len(SRC_FIELD.vocab)}")
    print(f"Target language vocab size: {len(TRG_FIELD.vocab)}")

    train_data, valid_data, test_data = data.split([0.8, 0.1, 0.1])

    print(f"Amount of train examples: {len(train_data.examples)}")
    print(f"Amount of valid examples: {len(valid_data.examples)}")
    print(f"Amount of test examples: {len(test_data.examples)}")

    return train_data, valid_data, test_data


def prepare_iterators(train_data, test_data, valid_data, batch_size, device):
    return BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device,
        sort=False
    )


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        prediction = model(src, trg)

        prediction = prediction[1:].view(-1, prediction.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(prediction, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
