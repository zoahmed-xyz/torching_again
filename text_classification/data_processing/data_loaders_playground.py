import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader


def get_news_data():
    train_iter = iter(ag_news_data)
    # take only the text part of the data
    return (line[1] for line in train_iter)


def create_newsdata_dataloader():
    ag_news_data = AG_NEWS(split='train')
    train_iter = iter(ag_news_data)
    train_dataloader = DataLoader(ag_news_data, batch_size=1, shuffle=True)
    # take only the text part of the data
    return train_dataloader


def get_news_data_from_dataloader():
    train_dataloader = create_newsdata_dataloader()
    # return only the text part from the train_dataloader as an iterable
    return (line[1][0] for line in train_dataloader)


def create_vocab_from_dataloader():
    train_dataloader = create_newsdata_dataloader()
    tokenizer = get_tokenizer('basic_english')
    # create a vocab from the data
    vocab = build_vocab_from_iterator(map(tokenizer, get_news_data_from_dataloader()), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def test_create_newsdata_dataloader():
    train_dataloader = create_newsdata_dataloader()
    for i, (label, text) in enumerate(train_dataloader):
        print("label: ", label)
        print("text: ", text)
        if i == 0:
            break
        if i == 100:
            break


# create a tokenizer
def create_vocab():
    tokenizer = get_tokenizer('basic_english')
    # create a vocab from the data
    vocab = build_vocab_from_iterator(map(tokenizer, get_news_data()), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def test_create_vocab_on_some_text(text):
    tokenizer = get_tokenizer('basic_english')
    vocab = create_vocab()
    print(vocab(tokenizer(text)))


def test_create_vocab_from_dataloader(text):
    tokenizer = get_tokenizer('basic_english')
    vocab = create_vocab_from_dataloader()
    print(vocab(tokenizer(text)))


def data_pipeline():
    vocab = create_vocab()
    tokenizer = get_tokenizer('basic_english')
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1
    return text_pipeline, label_pipeline


# write main function to test the code
if __name__ == '__main__':
    test_create_vocab_from_dataloader("Hello World")
