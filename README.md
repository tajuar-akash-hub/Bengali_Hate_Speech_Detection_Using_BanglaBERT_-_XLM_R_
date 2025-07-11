# Hate_Speech_Detection

## Here's the All Documention of my Implementaiton 

### **[BanglaBERT Documentation](https://github.com/tajuar-akash-hub/Bengali_Hate_Speech_Detection_Using_BanglaBERT_-_XLM_R_/blob/main/BanglaBERT.md)**


# Questions and Answers from BanglaBERT

## 1. What is `normalizer`?
The `normalizer` is a utility library specifically designed for text normalization in Bengali. It:
- Removes unnecessary characters or symbols.
- Fixes formatting issues like extra spaces.
- Ensures the text is consistent and ready for processing by models like BanglaBERT.

By using the `normalizer`, preprocessing becomes more efficient, improving tokenization and subsequent model performance.

---

## 2. What is a `wheel` in Python libraries?
A `wheel` is a distribution format used for Python packages. It is a pre-built package that can be installed without the need to compile code, making installations faster and more reliable. When you see "Building wheel" during installation, it means the library is being built into this format.

---

## 3. Why is `tokenization` important?
Tokenization is the process of splitting text into smaller units, such as words, subwords, or characters. For BanglaBERT:
- Converts raw text into numerical inputs that the model can process.
- Handles out-of-vocabulary words through subword tokenization.
- Ensures uniform input size via padding and truncation.

Tokenization aligns the input text with the model's pretrained vocabulary, significantly improving its ability to understand and process the text.

---

## 4. What is `weight decay` in the trainer function?
Weight decay is a regularization technique used to prevent overfitting. It:
- Penalizes large weights by adding a term to the loss function.
- Encourages the model to prefer smaller weights, making it less complex and more generalizable.
- Is implemented in the `AdamW` optimizer, which combines weight decay with adaptive learning rates.

---

## 5. What is a `utility function`?
Utility functions are helper functions that perform specific tasks to simplify the main operations of a program. In BanglaBERT, utility functions include:
- Data loading and preprocessing.
- Tokenization of text.
- Computing evaluation metrics (e.g., accuracy, F1-score).
- Saving and loading models or checkpoints.

These functions make the codebase modular and easier to maintain.

---

## 6. What is a `checkpoint`?
A checkpoint is a saved state of the model during training. It includes:
- Model weights and biases.
- Optimizer state.
- Learning rate scheduler state.

**Purpose**:
- Resume training from a specific point if interrupted.
- Restore the best-performing model for evaluation or deployment.

---

## 7. What is `early stopping`?
Early stopping is a training strategy that halts training when the model's performance on the validation set stops improving for a predefined number of epochs. **Benefits**:
- Prevents overfitting.
- Saves computational resources by stopping unnecessary training.

---

## 8. What is a `checkpoint 20, 30`?
These refer to checkpoints saved at the 20th and 30th epochs during training. These snapshots allow:
- Analysis of the model's performance at specific stages.
- Resumption of training from those points if needed.

---

## 9. What is `tokenizer_config.json`?
This file contains metadata for the tokenizer, including:
- Tokenizer type (e.g., WordPiece, Byte-Pair Encoding).
- Vocabulary size.
- Special tokens like `[PAD]`, `[CLS]`, and `[SEP]`.

---

## 10. What is `config.json`?
This file defines the model's architecture and settings, such as:
- Number of layers, hidden units, and attention heads.
- Training hyperparameters (e.g., learning rate, weight decay).
- Metadata about the model's purpose and source.

---

## 11. What is `sentencepiece.bpe.model`?
This file is part of the SentencePiece tokenizer, implementing Byte-Pair Encoding (BPE). It contains:
- Vocabulary and subword units.
- Rules for splitting text into subwords.

---

## 12. What is `tokenizer.json`?
This file provides a full representation of the tokenizer, including:
- Token-to-ID mappings.
- Encoding and decoding rules.
- Additional metadata for tokenization.

---

## 13. What is `vocab.txt`?
This file stores the vocabulary of the tokenizer. It includes:
- Tokens, subword units, or words.
- Their corresponding indices for numerical representation.
- Special tokens like `[PAD]` and `[UNK]`.

---

## 14. What is `model.safetensors`?
This file stores the trained model's weights in a safer and more efficient format compared to `.bin`. **Advantages**:
- Faster loading times.
- Reduced risk of file corruption.
- Better compatibility for distributed systems.

---

## 15. How to save models and what is saved?
Models can be saved using the `.save_pretrained()` method from the Hugging Face library. This saves:
1. **Model weights**: Parameters learned during training.
2. **Configuration**: Stored in `config.json`, which defines the model's architecture.
3. **Tokenizer**: Files like `tokenizer.json`, `tokenizer_config.json`, and `vocab.txt`.
4. **Special files**: Includes `model.safetensors` or `pytorch_model.bin` for weight storage.

**Example**:
```python
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
```

---

## 16. What happens when a model is saved to Hugging Face?
When saving a model to Hugging Face's hub, the following files are uploaded:
- **`config.json`**: Model architecture and metadata.
- **`tokenizer.json`**: Tokenizer rules and mappings.
- **`vocab.txt`**: Vocabulary of the tokenizer.
- **`model.safetensors` or `pytorch_model.bin`**: Trained weights of the model.

The model is then accessible via the Hugging Face hub, enabling easy sharing and reuse.

---

