import argparse
import os

from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer


def run_dataprep(normalizer: Normalizer) -> None:
    """
    Run the data preparation pipeline on the training corpus.

    Parameters:
        normalizer: A Normalizer instance.

    Returns:
        None.
    """
    train_raw_dir = os.getenv("TRAIN_RAW_DIR")
    train_tokens_path = os.getenv("TRAIN_TOKENS")

    raw_text = normalizer.load(train_raw_dir)
    raw_text = normalizer.strip_gutenberg(raw_text)

    sentences = normalizer.sentence_tokenize(raw_text)
    sentences = sentences[:100]

    tokenized_sentences = []

    for sentence in sentences:
        normalized_sentence = normalizer.normalize(sentence)

        if normalized_sentence:
            tokens = normalizer.word_tokenize(normalized_sentence)

            if tokens:
                tokenized_sentences.append(tokens)

    normalizer.save(tokenized_sentences, train_tokens_path)
    print(f"Saved tokenized training data to: {train_tokens_path}")


def main():
    """Parse command-line arguments and run the selected pipeline step."""
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser(
        description="N-gram next-word prediction system."
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=["dataprep", "model", "inference", "all"],
        help="Choose which pipeline step to run."
    )

    args = parser.parse_args()

    normalizer = Normalizer()

    if args.step == "dataprep":
        run_dataprep(normalizer)
    elif args.step == "model":
        print("Model step selected.")
    elif args.step == "inference":
        print("Inference step selected.")
    elif args.step == "all":
        run_dataprep(normalizer)
        print("Model step selected.")
        print("Inference step selected.")


if __name__ == "__main__":
    main()