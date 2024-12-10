from experimentPipeline import *
from nGramModel import *
from nGramRandomTextGenerator import *
from hiddenMarkovModelTextGenerator import *
from nanoGPTModel import *

if __name__ == '__main__':
    vocab = ['A', 'B', 'C', 'D']
    nGram = NGramModel(n=2, random_seed=42)
    info_source = nGramRandomTextGenerator(vocab=vocab, n=2, random_seed=42)
    # pipeline(info_source, nGram, source_length=100000)

    print(info_source.generate_text())
    print(info_source.calculate_perplexity())

    states = ["s1","s2","s3"]
    observations = ["A", "B"]
    state_prob = [1,0,0]
    transition_prob = [[0,1,0],
                       [0,0,1],
                       [1,0,0]]
    emission_prob = [[1,0],
                     [0,1],
                     [0,1]]
    hmm = hiddenMarkovModelTextGenerator(states, observations,state_prob=state_prob, transition_prob=transition_prob, emission_prob=emission_prob)
    # print(hmm.get_state_probability())
    # print(hmm.get_transition_probability())
    # print(hmm.get_emission_probability())
    # print(hmm.calculate_stationary_distribution())
    # print("vgvg")
    # print(hmm.generate_text())
    # print(hmm.calculate_perplexity())

    print(hmm.generate_text())
    pipeline(info_source, nGram, source_length=100000)

    nanoGpt = NanoGPT("/Users/arielsmog/PycharmProjects/information_theoretic_limits_of_large_language_models/nanoGPT-master/config/eval_gpt2.py",
                      "/Users/arielsmog/PycharmProjects/information_theoretic_limits_of_large_language_models/nanoGPT-master/config/train_shakespeare_char.py")
    nanoGpt.train("data/train.bin")
    print(nanoGpt)