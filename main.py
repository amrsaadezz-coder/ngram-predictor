import argparse
import os

from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel


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


def run_model(model: NGramModel) -> None:
    """
    Run the model-building pipeline on tokenized training data.

    Parameters:
        model: A preconfigured NGramModel instance.

    Returns:
        None.
    """
    train_tokens_path = os.getenv("TRAIN_TOKENS")
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")

    sentences = model.load_tokenized_sentences(train_tokens_path)
    model.build_vocab(sentences)
    updated_sentences = model.replace_rare_words(sentences)
    model.build_counts_and_probabilities(updated_sentences)
    model.save_model(model_path, vocab_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved vocabulary to: {vocab_path}")


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

    ngram_order = int(os.getenv("NGRAM_ORDER"))
    unk_threshold = int(os.getenv("UNK_THRESHOLD"))

    normalizer = Normalizer()
    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold)

    if args.step == "dataprep":
        run_dataprep(normalizer)
    elif args.step == "model":
        run_model(model)
    elif args.step == "inference":
        print("Inference step selected.")
    elif args.step == "all":
        run_dataprep(normalizer)
        run_model(model)
        print("Inference step selected.")


if __name__ == "__main__":
    main()