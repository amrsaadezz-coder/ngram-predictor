# N-Gram Next-Word Predictor

A beginner-friendly next-word prediction system built from scratch using an n-gram language model with backoff. The project trains on Sherlock Holmes novels, builds n-gram probability tables, and predicts the most likely next word from user input through a command-line interface.

## Requirements

- Python 3.11
- Anaconda
- Install dependencies from `requirements.txt`

## Setup

1. Clone the repository.
2. Create and activate an Anaconda environment.
3. Install dependencies using `requirements.txt`.
4. Create and populate `config/.env`.
5. Place the four training `.txt` files inside `data/raw/train/`.
6. Optional: place one evaluation `.txt` file inside `data/raw/eval/`.

## Usage

Run each step from the project root:

```bash
python main.py --step dataprep
python main.py --step model
python main.py --step inference
python main.py --step all
```

## Project Structure

```text
ngram-predictor/
├── config/
│   └── .env
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── eval/
│   ├── processed/
│   └── model/
├── src/
│   ├── data_prep/
│   │   └── normalizer.py
│   ├── model/
│   │   └── ngram_model.py
│   ├── inference/
│   │   └── predictor.py
│   ├── ui/
│   │   └── app.py
│   └── evaluation/
│       └── evaluator.py
├── main.py
├── tests/
│   ├── test_data_prep.py
│   ├── test_model.py
│   ├── test_inference.py
│   ├── test_ui.py
│   └── test_evaluation.py
├── .gitignore
├── requirements.txt
└── README.md
```