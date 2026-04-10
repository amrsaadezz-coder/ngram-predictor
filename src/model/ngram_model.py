import json
from collections import Counter, defaultdict
from typing import Dict, List


class NGramModel:
    """Build, store, load, and query n-gram probability tables."""

    def __init__(self, ngram_order: int, unk_threshold: int):
        """
        Initialize the n-gram model.

        Parameters:
            ngram_order: Maximum n-gram order to build.
            unk_threshold: Minimum frequency required for a word to stay in vocabulary.

        Returns:
            None.
        """
        self.ngram_order = ngram_order
        self.unk_threshold = unk_threshold
        self.vocab = set()
        self.model = {}
        self.ngram_counts = {}
        self.word_counts = Counter()

    def load_tokenized_sentences(self, filepath: str) -> List[List[str]]:
        """
        Load tokenized sentences from a file.

        Parameters:
            filepath: Path to the tokenized sentence file.

        Returns:
            A list of tokenized sentences.
        """
        sentences = []

        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                tokens = line.strip().split()
                if tokens:
                    sentences.append(tokens)

        return sentences

    def build_vocab(self, sentences: List[List[str]]) -> set:
        """
        Build the vocabulary from tokenized sentences using the UNK threshold.

        Parameters:
            sentences: A list of tokenized sentences.

        Returns:
            A set containing the vocabulary words plus <UNK>.
        """
        self.word_counts = Counter()

        for sentence in sentences:
            self.word_counts.update(sentence)

        vocab = {
            word for word, count in self.word_counts.items()
            if count >= self.unk_threshold
        }
        vocab.add("<UNK>")

        self.vocab = vocab
        return vocab

    def replace_rare_words(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Replace words not in the vocabulary with <UNK>.

        Parameters:
            sentences: A list of tokenized sentences.

        Returns:
            A new list of tokenized sentences with rare words replaced by <UNK>.
        """
        updated_sentences = []

        for sentence in sentences:
            updated_sentence = [
                word if word in self.vocab else "<UNK>"
                for word in sentence
            ]
            updated_sentences.append(updated_sentence)

        return updated_sentences

    def build_counts_and_probabilities(self, sentences: List[List[str]]) -> Dict[str, Dict]:
        """
        Build n-gram counts and MLE probabilities for all orders.

        Parameters:
            sentences: A list of tokenized sentences.

        Returns:
            A dictionary containing probability tables for all n-gram orders.
        """
        ngram_counts = {
            order: Counter() for order in range(1, self.ngram_order + 1)
        }

        for sentence in sentences:
            for order in range(1, self.ngram_order + 1):
                if len(sentence) >= order:
                    for index in range(len(sentence) - order + 1):
                        ngram = tuple(sentence[index:index + order])
                        ngram_counts[order][ngram] += 1

        self.ngram_counts = ngram_counts

        model = {}

        total_unigrams = sum(ngram_counts[1].values())

        unigram_probs = {}
        for ngram, count in ngram_counts[1].items():
            unigram_probs[ngram[0]] = count / total_unigrams

        model["1gram"] = unigram_probs

        for order in range(2, self.ngram_order + 1):
            order_probs = defaultdict(dict)

            for ngram, count in ngram_counts[order].items():
                prefix = ngram[:-1]
                next_word = ngram[-1]
                prefix_count = ngram_counts[order - 1][prefix]

                if prefix_count > 0:
                    order_probs[" ".join(prefix)][next_word] = count / prefix_count

            model[f"{order}gram"] = dict(order_probs)

        self.model = model
        return model

    def save_model(self, model_path: str, vocab_path: str) -> None:
        """
        Save the model probabilities and vocabulary to JSON files.

        Parameters:
            model_path: Path to save model.json.
            vocab_path: Path to save vocab.json.

        Returns:
            None.
        """
        with open(model_path, "w", encoding="utf-8") as model_file:
            json.dump(self.model, model_file, indent=2)

        with open(vocab_path, "w", encoding="utf-8") as vocab_file:
            json.dump(sorted(self.vocab), vocab_file, indent=2)

    def load(self, model_path: str, vocab_path: str) -> None:
        """
        Load the model probabilities and vocabulary from JSON files.

        Parameters:
            model_path: Path to model.json.
            vocab_path: Path to vocab.json.

        Returns:
            None.
        """
        with open(model_path, "r", encoding="utf-8") as model_file:
            self.model = json.load(model_file)

        with open(vocab_path, "r", encoding="utf-8") as vocab_file:
            self.vocab = set(json.load(vocab_file))

    def lookup(self, context: List[str]) -> Dict[str, float]:
        """
        Look up next-word probabilities using backoff from highest order to unigram.

        Parameters:
            context: The input context words.

        Returns:
            A dictionary mapping candidate next words to probabilities.
        """
        for order in range(self.ngram_order, 1, -1):
            required_context_length = order - 1
            reduced_context = context[-required_context_length:]
            context_key = " ".join(reduced_context)

            if context_key in self.model.get(f"{order}gram", {}):
                return self.model[f"{order}gram"][context_key]

        return self.model.get("1gram", {})


def main():
    """Run the n-gram model module in isolation."""
    print("NGramModel module is ready for testing.")


if __name__ == "__main__":
    main()