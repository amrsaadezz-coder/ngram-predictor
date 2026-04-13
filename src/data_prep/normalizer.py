import os
import re
from typing import List

import nltk


class Normalizer:
    """Load, clean, normalize, tokenize, and save corpus text."""

    def load(self, folder_path: str) -> str:
        """
        Load all .txt files from a folder, strip Gutenberg boilerplate
        from each file, and concatenate them into one text string.

        Parameters:
            folder_path: Path to the folder containing text files.

        Returns:
            A single string containing the cleaned contents of all .txt files.
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Training folder not found: {folder_path}. "
                "Please create it and place the raw .txt books inside."
            )

        texts = []

        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)

                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()

                text = self.strip_gutenberg(text)
                texts.append(text)

        return "\n".join(texts)

    def strip_gutenberg(self, text: str) -> str:
        """
        Remove Project Gutenberg header and footer from a text.

        Parameters:
            text: Raw text content.

        Returns:
            Text with Gutenberg header and footer removed where possible.
        """
        start_match = re.search(
            r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        end_match = re.search(
            r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        if start_match:
            text = text[start_match.end():]

        if end_match:
            text = text[:end_match.start()]

        return text.strip()

    def lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.

        Parameters:
            text: Input text.

        Returns:
            Lowercased text.
        """
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.

        Parameters:
            text: Input text.

        Returns:
            Text with punctuation removed.
        """
        return re.sub(r"[^a-z\s]", " ", text)

    def remove_numbers(self, text: str) -> str:
        """
        Remove digits from text.

        Parameters:
            text: Input text.

        Returns:
            Text with numbers removed.
        """
        return re.sub(r"\d+", "", text)

    def remove_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace and blank lines.

        Parameters:
            text: Input text.

        Returns:
            Cleaned text with normalized spacing.
        """
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize(self, text: str) -> str:
        """
        Apply the full normalization pipeline in the required order.

        Parameters:
            text: Input text.

        Returns:
            Normalized text.
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Parameters:
            text: Input text.

        Returns:
            A list of sentence strings.
        """
        return nltk.sent_tokenize(text)

    def word_tokenize(self, sentence: str) -> List[str]:
        """
        Split a sentence into word tokens.

        Parameters:
            sentence: A single sentence.

        Returns:
            A list of word tokens.
        """
        return sentence.split()

    def save(self, sentences: List[List[str]], filepath: str) -> None:
        """
        Save tokenized sentences to a file, one sentence per line.

        Parameters:
            sentences: List of tokenized sentences.
            filepath: Output file path.

        Returns:
            None.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as file:
            for sentence in sentences:
                file.write(" ".join(sentence) + "\n")


def main():
    """Run the normalizer module in isolation."""
    print("Normalizer module is ready for testing.")


if __name__ == "__main__":
    main()