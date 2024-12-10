import numpy as np
import os
import tiktoken

def create_dataset(information_source, length):
    input_file_path = os.path.join('/Users/arielsmog/PycharmProjects/information_theoretic_limits_of_large_language_models/data/input.txt')
    text = information_source.generate_text(length=length)
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    data = text
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile('/Users/arielsmog/PycharmProjects/information_theoretic_limits_of_large_language_models/data/train.bin')
    val_ids.tofile('/Users/arielsmog/PycharmProjects/information_theoretic_limits_of_large_language_models/data/val.bin')


def pipeline(information_source, language_model, source_length):
    create_dataset(information_source, source_length)
    with open('/Users/arielsmog/PycharmProjects/information_theoretic_limits_of_large_language_models/data/input.txt', 'r') as f:
        text = f.read()
    language_model.train(text)
    print(f"the perplexity of the information source is {information_source.calculate_perplexity()}, "
          f"and the perplexity of the model is {language_model.calculate_perplexity()}")