from hmmlearn import hmm
import numpy as np
from scipy.linalg import eig
import math

class hiddenMarkovModelTextGenerator:
    def __init__(self, states, observations, state_prob=None, transition_prob=None, emission_prob=None, random_seed=None):
        self.num_states = len(states)
        self.num_observations = len(observations)
        self.states = states
        self.observations = observations
        self.state_probability = state_prob
        self.transition_probability = transition_prob
        self.emission_probability = emission_prob
        if random_seed is not None:
            np.random.seed(random_seed)

        if state_prob is None:
            self.random_state_prob()
        if transition_prob is None:
            self.random_transition_prob()
        if emission_prob is None:
            self.random_emission_prob()

        self.model = hmm.CategoricalHMM(n_components=self.num_states)
        self.model.startprob_ = self.state_probability
        self.model.transmat_ = self.transition_probability
        self.model.emissionprob_ = self.emission_probability

    def random_state_prob(self):
        random_probs = [np.random.random() for _ in range(self.num_states)]
        total = sum(random_probs)
        normalized_probs = [p / total for p in random_probs]
        self.state_probability = normalized_probs

    def random_transition_prob(self):
        transition_probs = []
        for i in range(self.num_states):
            random_probs = [np.random.random() for _ in range(self.num_states)]
            total = sum(random_probs)
            normalized_probs = [p / total for p in random_probs]
            transition_probs.append(normalized_probs)
        self.transition_probability = transition_probs

    def random_emission_prob(self):
        emission_probs = []
        for i in range(self.num_states):
            random_probs = [np.random.random() for _ in range(self.num_observations)]
            total = sum(random_probs)
            normalized_probs = [p / total for p in random_probs]
            emission_probs.append(normalized_probs)
        self.emission_probability = emission_probs

    def generate_text(self, length=10):
        text = self.model.sample(length)[0]
        text = text.tolist()
        text = [self.observations[i[0]] for i in text]
        text = ' '.join(text)
        return text

    def get_state_probability(self):
        return self.state_probability
    def get_transition_probability(self):
        return self.transition_probability
    def get_emission_probability(self):
        return self.emission_probability

    def calculate_stationary_distribution(self):
        """Calculates the stationary distribution from the transition matrix."""
        eigvals, eigvecs = eig(np.array(self.transition_probability).T)
        stationary = np.array(eigvecs[:, np.isclose(eigvals, 1)])
        stationary = stationary / stationary.sum()  # Normalize to make it a probability distribution
        stationary = stationary.real.flatten()  # Take the real part
        return stationary

    def calculate_entropy_rate(self):
        """Calculates the entropy rate using the stationary distribution and transition matrix."""
        stationary_distribution = self.calculate_stationary_distribution()
        entropy_rate = 0.0
        for i, pi in enumerate(stationary_distribution):
            for j, p_ij in enumerate(self.transition_probability[i]):
                if p_ij > 0:  # Avoid log(0)
                    entropy_rate += pi * p_ij * (-math.log2(p_ij))
        return entropy_rate

    def calculate_perplexity(self):
        """Calculates the perplexity based on entropy rate."""
        entropy_rate = self.calculate_entropy_rate()
        perplexity = 2 ** entropy_rate
        return perplexity