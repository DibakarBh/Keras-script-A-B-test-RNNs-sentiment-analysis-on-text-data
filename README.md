# RNN Sentiment Analysis: Scratch vs. GloVe Embeddings

This repository contains a comparative study of Word Embedding strategies for Sentiment Analysis. It explores the performance gap between Bidirectional LSTMs using pre-trained GloVe vectors and those trained with task-specific embeddings from scratch.

## Project Structure
The experiment is designed as a controlled benchmark to observe how model capacity influences performance across varying dataset sizes:

- **Model A (Scratch)**: Fully trainable 100d Embedding layer (~1.03M params).
- **Model B (GloVe)**: Frozen 100d GloVe embeddings (~34k trainable params).

## Key Features
- **Bidirectional LSTM Stack**: Captures temporal dependencies in both directions for nuanced sentiment detection.
- **Dynamic Training Subsets**: Iterative testing across sample sizes (100 to 20,000) to locate the performance "crossover point."
- **Robust Preprocessing**: Custom standardization and vocabulary adaptation on the full dataset to ensure completely fair comparison.

## Experimental Results
- **100 Samples**: The "Scratch" model reached 1.00 training accuracy (overfitting) but achieved a higher test accuracy of 62.45% compared to GloVe's 57.64%.
- **Crossover Point**: A statistical crossover was identified at **250 samples**, where the Scratch model's learning capacity consistently surpassed the frozen pre-trained baseline.
- **Complexity Analysis**: Demonstrates that while GloVe provides stability, task-specific training can capture niche data features more effectively given sufficient (even if small) data.

## Tech Stack
- **Framework**: TensorFlow / Keras
- **Architecture**: Bidirectional LSTMs
- **NLP**: GloVe (Global Vectors for Word Representation)
