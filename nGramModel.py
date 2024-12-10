import random

import nltk_punkt
import numpy as np
from collections import Counter, defaultdict
from nltk import ngrams
import nltk
import math
import itertools
from scipy.linalg import eig
# nltk.download('punkt_tab')

class NGramModel:
    def __init__(self, n=2, random_seed=None):
        """
        Initializes the NGramModel.

        :param n: The 'n' in n-gram, e.g., 2 for bigram, 3 for trigram, etc.
        """
        self.n = n
        self.ngram_freqs = defaultdict(lambda: defaultdict(int))
        self.vocab_distribution = {}

        if random_seed is not None:
            random.seed(random_seed)
            self.random_seed = random_seed

    def get_ngram_freqs(self):
        return self.ngram_freqs

    def get_vocab_distribution(self):
        return self.vocab_distribution

    def train(self, text, smoothing=False):
      """
      Calculates the n-gram distribution from the given text with optional Add-1 smoothing.

      :param text: Input text to build the n-gram model.
      :param smoothing: Whether to apply Add-1 smoothing (default is False).
      """
      # Tokenize the text
      tokens = nltk.word_tokenize(text.lower())

      # Generate n-grams
      n_grams = list(ngrams(tokens, self.n))

      # Count n-gram frequencies
      ngram_counts = Counter(n_grams)

      # Calculate the total frequency for each prefix
      prefix_counts = defaultdict(int)
      for ngram, freq in ngram_counts.items():
          prefix = ngram[:-1]
          self.ngram_freqs[prefix][ngram[-1]] = freq
          prefix_counts[prefix] += freq

      # Get vocabulary
      vocab = set(tokens)
      vocab_size = len(vocab)

      # Convert counts to probabilities
      for prefix, suffix_counts in self.ngram_freqs.items():
          total_count = prefix_counts[prefix]
          if smoothing:
              # Apply Add-1 smoothing
              total_count += vocab_size  # Adjust total count
              self.vocab_distribution[prefix] = {
                  suffix: (count + 1) / total_count  # Add 1 to each count
                  for suffix, count in suffix_counts.items()
              }
              # Account for unseen suffixes
              for unseen_suffix in vocab - suffix_counts.keys():
                  self.vocab_distribution[prefix][unseen_suffix] = 1 / total_count
          else:
              # No smoothing
              self.vocab_distribution[prefix] = {
                  suffix: count / total_count
                  for suffix, count in suffix_counts.items()
              }

    def generate_text(self, seed, length=10):
        """
        Generates text by sampling from the n-gram distribution.

        :param seed: Seed text to start generating from.
        :param length: Number of tokens to generate.
        :return: Generated text as a string.
        """
        if not self.vocab_distribution:
            raise ValueError("The n-gram distribution has not been calculated yet.")


        # Tokenize the seed
        seed_tokens = nltk.word_tokenize(seed.lower())

        # If the seed is shorter than n-1, extend it with a random starting point from the n-gram model
        if len(seed_tokens) < (self.n - 1):
            # Choose a random prefix (starting point) from the vocab distribution
            current_prefix = random.choice(list(self.vocab_distribution.keys()))
            generated_tokens = list(seed_tokens)
        else:
            # Start with the last n-1 characters of the seed
            current_prefix = tuple(seed_tokens[-(self.n - 1):])
            generated_tokens = list(seed_tokens)


        if self.n > 1:
          for _ in range(length):
              if current_prefix in self.vocab_distribution:
                  # Sample next word based on current prefix distribution
                  next_word = random.choices(
                      list(self.vocab_distribution[current_prefix].keys()),
                      list(self.vocab_distribution[current_prefix].values())
                  )[0]
                  generated_tokens.append(next_word)
                  current_prefix = tuple(generated_tokens[-(self.n - 1):])
              else:
                  # Stop if no continuation is found
                  break
        else:
          for _ in range(length):
            next_word = random.choices(
                      list(self.vocab_distribution[()].keys()),
                      list(self.vocab_distribution[()].values()))[0]
            generated_tokens.append(next_word)

        return ' '.join(generated_tokens)

    def build_transition_matrix(self):
        """Builds the transition matrix for the prefixes."""

        prefixes = list(self.vocab_distribution.keys())
        prefix_index = {prefix: idx for idx, prefix in enumerate(prefixes)}
        size = len(prefixes)

        # Initialize transition matrix
        transition_matrix = np.zeros((size, size))

        for i, prefix in enumerate(prefixes):
            for suffix, prob in self.vocab_distribution[prefix].items():
                next_prefix = tuple(list(prefix[1:]) + [suffix])
                if next_prefix in prefix_index:
                    j = prefix_index[next_prefix]
                    transition_matrix[i, j] = prob

        return transition_matrix, prefixes

    def calculate_stationary_distribution(self):
        """Solves for the stationary distribution using the transition matrix."""
        transition_matrix, prefixes = self.build_transition_matrix()

        # Find the eigenvector associated with eigenvalue 1
        eigvals, eigvecs = eig(transition_matrix.T)
        stationary = np.array(eigvecs[:, np.isclose(eigvals, 1)])

        # Normalize to make it a probability distribution
        stationary = stationary / stationary.sum()
        stationary = stationary.real.flatten()  # Take the real part



        # Map back to prefixes
        stationary_distribution = {prefix: stationary[i] for i, prefix in enumerate(prefixes)}
        return stationary_distribution


    def calculate_entropy_rate(self):
        """Calculates the entropy rate using stationary distribution and transition probabilities."""
        # Estimate the stationary distribution
        stationary_distribution = self.calculate_stationary_distribution()

        entropy_rate = 0.0
        for prefix, prefix_prob in stationary_distribution.items():
            if prefix in self.vocab_distribution:
                for suffix, transition_prob in self.vocab_distribution[prefix].items():
                    entropy_rate += prefix_prob * transition_prob * (-math.log2(transition_prob))

        return entropy_rate

    def calculate_perplexity(self):
      """calculates the perplexity of the text based on the relation between perplexity and entropy rate"""
      entropy_rate = self.calculate_entropy_rate()
      return 2 ** entropy_rate
