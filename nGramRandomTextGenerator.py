import random
import itertools
import numpy as np
import math
import itertools
from scipy.linalg import eig

class nGramRandomTextGenerator:
  def __init__(self, vocab, n=2, random_seed=None):
    self.n = n
    self.vocab = set(vocab)
    self.vocab_distribution = {}

    if random_seed is not None:
      random.seed(random_seed)

  def create_random_distribution(self, smoothing=False):
      """
      Creates a random n-gram distribution over the given vocabulary.
      :param smoothing: Whether to apply Add-1 smoothing to the random probabilities.
      """
      vocab_size = len(self.vocab)

      # Create random distributions for each possible n-gram prefix
      prefixes = itertools.product(self.vocab, repeat=self.n - 1) if self.n > 1 else [()]
      for prefix in prefixes:
          # Generate random probabilities for each word in the vocabulary
          random_probs = [random.random() for _ in self.vocab]
          total = sum(random_probs)
          normalized_probs = [p / total for p in random_probs]

          # Assign probabilities to the distribution
          self.vocab_distribution[prefix] = {
              word: prob for word, prob in zip(self.vocab, normalized_probs)
          }

          if smoothing:
              # Apply Add-1 smoothing to the random distribution
              smoothed_total = sum(prob + 1 for prob in random_probs) + vocab_size
              self.vocab_distribution[prefix] = {
                  word: (prob + 1) / smoothed_total
                  for word, prob in zip(self.vocab, random_probs)
              }

  def generate_text(self, length=10, smoothing=False):
      """
      Generates text by sampling from the random n-gram distribution.
      :param length: Number of tokens to generate.
      :return: Generated text as a string.
      """
      if not self.vocab_distribution:
          self.create_random_distribution(smoothing=False)

      if not self.vocab:
          raise ValueError("The vocabulary is empty. Ensure the distribution is set up correctly.")

      # Sample initial prefix randomly from the vocabulary
      vocab_list = list(self.vocab)
      current_prefix = tuple(random.choices(vocab_list, k=self.n - 1)) if self.n > 1 else ()
      generated_tokens = list(current_prefix)

      # Generate the remaining tokens
      for _ in range(length - len(current_prefix)):
          if current_prefix in self.vocab_distribution:
              # Sample next word based on current prefix distribution
              next_word = random.choices(
                  list(self.vocab_distribution[current_prefix].keys()),
                  list(self.vocab_distribution[current_prefix].values())
              )[0]
              generated_tokens.append(next_word)
              # Update the current prefix
              current_prefix = tuple(generated_tokens[-(self.n - 1):]) if self.n > 1 else ()
          else:
              # Stop if no continuation is found
              break

      return ' '.join(generated_tokens)

  def get_vocab_distribution(self):
      return self.vocab_distribution



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