from typing import List

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel


class Predictor:
    """Accept a pre-loaded model and normalizer, prepare input text, and return top-k next-word predictions."""

    def __init__(self, model: NGramModel, normalizer: Normalizer):
        """
        Initialize the predictor with dependencies created elsewhere.

        Parameters:
            model: A pre-loaded NGramModel instance.
            normalizer: A Normalizer instance.

        Returns:
            None.
        """
        self.model = model
        self.normalizer = normalizer

    def normalize(self, text: str) -> List[str]:
        """
        Normalize input text and extract the last n-1 words as context.

        Parameters:
            text: User input text.

        Returns:
            A list of context words.
        """
        normalized_text = self.normalizer.normalize(text)
        words = normalized_text.split()

        context_length = self.model.ngram_order - 1
        return words[-context_length:] if context_length > 0 else words

    def map_oov(self, context: List[str]) -> List[str]:
        """
        Replace out-of-vocabulary words in the context with <UNK>.

        Parameters:
            context: A list of context words.

        Returns:
            A list of context words with OOV words mapped to <UNK>.
        """
        return [
            word if word in self.model.vocab else "<UNK>"
            for word in context
        ]

    def predict_next(self, text: str, k: int) -> List[str]:
        """
        Predict the top-k next words for the given input text.

        Parameters:
            text: User input text.
            k: Number of top predictions to return.

        Returns:
            A list of predicted next words sorted by probability.
        """
        context = self.normalize(text)
        context = self.map_oov(context)

        candidates = self.model.lookup(context)

        if not candidates:
            return []

        sorted_candidates = sorted(
            candidates.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return [word for word, _ in sorted_candidates[:k]]


def main():
    """Run the predictor module in isolation."""
    print("Predictor module is ready for testing.")


if __name__ == "__main__":
    main()