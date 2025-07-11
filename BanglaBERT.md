
# Bengali Hate Speech Detection Using BanglaBERT

This repository implements Bengali hate speech detection using the **BanglaBERT** pretrained model. The project focuses on classifying Bengali text into hate or non-hate speech using advanced natural language processing techniques, including data preprocessing, augmentation, and fine-tuning BanglaBERT.

---

## Contents
1. [Features](#features)
2. [Setup and Installation](#setup-and-installation)
3. [Code Explanation](#code-explanation)
4. [Trainer Details](#trainer-details)
5. [Results](#results)
6. [How to Run](#how-to-run)
7. [Future Improvements](#future-improvements)
8. [License](#license)

---

## Features
- **Text Preprocessing**: Includes normalization, cleaning, and tokenization for Bengali text.
- **Data Augmentation**: Adds techniques like synonym replacement, random insertion, and deletion to balance dataset classes.
- **Model Training**: Fine-tunes the BanglaBERT model with custom learning rates, dropout, and warm-up steps.
- **Evaluation**: Metrics include accuracy, precision, recall, F1-score, and AUC.
- **Prediction**: Predicts hate or non-hate speech with confidence scores.
- **Deployment**: Saves the trained model for reuse and deployment via Hugging Face's hub.

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tajuar-akash-hub/Bengali_Hate_Speech_Detection_Using_BanglaBERT_-_XLM_R_.git
   cd Bengali_Hate_Speech_Detection_Using_BanglaBERT_-_XLM_R_
   ```

2. Install dependencies:
   ```bash
   pip install normalizer transformers torch datasets pandas scikit-learn
   ```

3. Open the notebook or script and execute the code.

---

## Code Explanation

### Key Functions and Their Roles

#### 1. **`load_data()`**
   - **Purpose**: Attempts to load the hate speech dataset from predefined paths.
   - **Functionality**:
     - Tries to load the dataset from multiple file paths.
     - Raises a `FileNotFoundError` if the dataset is not found.

#### 2. **`enhanced_preprocess(text)`**
   - **Purpose**: Cleans and normalizes Bengali text.
   - **Functionality**:
     - Removes non-Bengali characters.
     - Replaces multiple spaces with a single space.
     - Applies BanglaBERT's normalizer for consistent formatting.

#### 3. **Data Augmentation Functions**
   - **`synonym_replace(text)`**: Replaces words with their synonyms to generate augmented text.
   - **`random_insertion(text)`**: Inserts random words into the text.
   - **`random_deletion(text)`**: Randomly deletes words from the text.

#### 4. **`balance_with_augmentation(df)`**
   - **Purpose**: Balances the dataset by augmenting minority class samples.
   - **Functionality**:
     - Generates new samples for the minority class using augmentation techniques.
     - Ensures the dataset is stratified.

#### 5. **`tokenize(examples)`**
   - **Purpose**: Tokenizes the text for input into the BanglaBERT model.
   - **Functionality**:
     - Uses the BanglaBERT tokenizer to encode the text with padding and truncation.

#### 6. **`compute_metrics(eval_pred)`**
   - **Purpose**: Evaluates the model's predictions.
   - **Functionality**:
     - Computes metrics such as accuracy, precision, recall, F1-score, and AUC.

#### 7. **`predict(text)`**
   - **Purpose**: Predicts whether a given text is hate speech or not.
   - **Functionality**:
     - Preprocesses and tokenizes the input text.
     - Uses the trained model to generate predictions and confidence scores.

#### 8. **`push_to_hf(username)`**
   - **Purpose**: Uploads the trained model to Hugging Face's hub.
   - **Functionality**:
     - Saves the model and tokenizer locally.
     - Pushes them to the specified Hugging Face repository.

---

## Trainer Details

The training process is handled by a custom trainer class that extends the Hugging Face `Trainer` class.

### Key Features of the Trainer
- **Logging**: Tracks training and evaluation metrics (e.g., loss, accuracy, F1-score) over epochs.
- **Custom Optimization**:
  - Implements a learning rate scheduler with warm-up steps.
  - Supports gradient accumulation for larger effective batch sizes.
- **Dropout Regularization**: Adds dropout layers to prevent overfitting.
- **Mixed Precision Training**: Utilizes GPU acceleration with `fp16` for faster training.

### Available Trainer Functions
1. **`log(logs)`**
   - Logs the training metrics for each epoch.
   - Stores the metrics for visualization.

2. **`create_optimizer()`**
   - Sets up custom learning rates for different layers of the model.
   - Uses the AdamW optimizer for weight decay regularization.

3. **`create_scheduler(num_training_steps)`**
   - Creates a learning rate scheduler with a linear warm-up phase.

### Training Hyperparameters
- Learning Rate: `3e-5`
- Batch Size: `16` (train), `32` (eval)
- Epochs: `15`
- Gradient Accumulation: `2` (effective batch size: `64`)

---

## Results

### Baseline Model
- **Accuracy**: `83.0%`
- **F1-Score**: `0.83`
- **AUC**: `0.868`

### Enhanced Model
- **Accuracy**: `93.4%`
- **F1-Score**: `0.934`
- **AUC**: `0.984`

### Visualizations
- Loss Curve: Tracks validation loss over epochs.
- Accuracy Curve: Tracks validation accuracy over epochs.
- Confusion Matrix: Highlights the model's performance on hate vs. non-hate classes.
- ROC Curve: Shows the trade-off between true positive rate (TPR) and false positive rate (FPR).

---

## How to Run

1. **Train the Model**
   Run the training script to fine-tune BanglaBERT on your dataset.

2. **Evaluate the Model**
   Use the evaluation script to compute metrics and visualize results.

3. **Make Predictions**
   Use the `predict` function to classify new text samples:
   ```python
   result = predict("তুমি একজন খারাপ মানুষ।")
   print(result)
   ```

4. **Deploy the Model**
   Push the trained model to Hugging Face for deployment:
   ```python
   push_to_hf("your_username")
   ```

---

## Future Improvements

- **Multilingual Support**: Incorporate code-mixed text handling.
- **Explainability**: Add methods to interpret model predictions.
- **Data Augmentation**: Experiment with advanced techniques like back-translation.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
