from experimentPipeline import *
from nGramModel import *
from nGramRandomTextGenerator import *
from hiddenMarkovModelTextGenerator import *

if __name__ == '__main__':
    vocab = ['A', 'B', 'C', 'D']
    biGram = NGramModel(n=2, random_seed=42)
    triGram = NGramModel(n=3, random_seed=42)
    # info_source = nGramRandomTextGenerator(vocab=vocab, n=2, random_seed=42)

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
    # print(hmm.generate_text())
    # print(hmm.calculate_perplexity())
    print("biGram:")
    pipeline(hmm, biGram, source_length=100000)
    print("triGram:")
    pipeline(hmm, triGram, source_length=100000)

