import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
import optuna # Used for the hyperparameter search

# --- 1. Configuration ---
# --- ✅ MODIFIED ---
TRAIN_FILE_PATH = '/content/sample_data/zho.csv'
TEST_FILE_PATH = '/content/sample_data/dev_zho.csv' # <-- See instructions
OUTPUT_FILE_PATH = 'predictions_for_evaluation.csv' # <-- This is your final file

ID_COLUMN = 'id'
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'polarization'
MODEL_NAME = 'roberta-large' # Use the powerful model

# Check if a GPU is available
if not torch.cuda.is_available():
    print("--- ⚠️ WARNING: No GPU detected. ---")
    print("This script is designed for a GPU. Please enable the T4 GPU in Colab.")
else:
    print("--- ✅ GPU Detected. Ready for training. ---")
    print(f"Device: {torch.cuda.get_device_name(0)}")


def load_and_prepare_data():
    """
    Loads BOTH the training and test data.
    """
    try:
        df_train_full = pd.read_csv(TRAIN_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Training file not found at {TRAIN_FILE_PATH}")
        print("Please make sure 'eng.csv' is uploaded to Colab.")
        return None, None

    try:
        df_test = pd.read_csv(TEST_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Test file not found at {TEST_FILE_PATH}")
        print("--- PLEASE UPLOAD YOUR UNLABELED TEST FILE ---")
        print("--- AND RENAME IT TO 'test_set_unlabeled.csv' ---")
        return None, None

    # --- 1. Prepare Training Data ---
    df_train_full = df_train_full.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df_train_full[LABEL_COLUMN] = df_train_full[LABEL_COLUMN].astype(int)
    df_train_full = df_train_full.rename(columns={LABEL_COLUMN: 'label', TEXT_COLUMN: 'text'})

    # We will split the training data to find the best hyperparameters
    train_df, eval_df = train_test_split(
        df_train_full, test_size=0.2, random_state=42, stratify=df_train_full['label']
    )

    # --- 2. Balance the (split) training set ---
    print("\n--- Balancing training data for hyperparameter search... ---")
    label_counts = train_df['label'].value_counts()
    if len(label_counts) < 2:
        train_df_balanced = train_df
    else:
        majority_label, minority_label = label_counts.idxmax(), label_counts.idxmin()
        max_count = label_counts.max()
        df_majority = train_df[train_df['label'] == majority_label]
        df_minority = train_df[train_df['label'] == minority_label]
        df_minority_upsampled = df_minority.sample(n=max_count, replace=True, random_state=42)
        train_df_balanced = pd.concat([df_majority, df_minority_upsampled])
        train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- 3. Create Hugging Face Datasets ---
    train_dataset = Dataset.from_pandas(train_df_balanced)
    eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))

    # This is our *final* training set (100% of eng.csv)
    full_train_dataset = Dataset.from_pandas(df_train_full.reset_index(drop=True))

    # This is our *final* test set (the 133 rows)
    # We must rename the 'text' column to match
    df_test = df_test.rename(columns={TEXT_COLUMN: 'text'})
    test_dataset = Dataset.from_pandas(df_test)

    print(f"Data loaded:")
    print(f"- {len(train_dataset)} samples for hyperparameter search training.")
    print(f"- {len(eval_dataset)} samples for hyperparameter search validation.")
    print(f"- {len(full_train_dataset)} samples for FINAL training.")
    print(f"- {len(test_dataset)} samples for FINAL prediction. (This should be 133)")

    return train_dataset, eval_dataset, full_train_dataset, test_dataset

def model_init():
    """Creates a new, fresh model for each search trial."""
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def tokenize_function(examples, tokenizer):
    """Tokenizes the text data."""
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def compute_metrics(eval_pred):
    """Computes Macro F1-Score for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average='macro')
    return {'macro_f1': macro_f1}

def main():
    train_dataset, eval_dataset, full_train_dataset, test_dataset = load_and_prepare_data()
    if train_dataset is None:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- 1. Tokenize all datasets ---
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_eval = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_full_train = full_train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # --- 2. Define Training Arguments (Modern) ---
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        metric_for_best_model="macro_f1",
        fp16=True, # GPU speedup
        report_to="none"
    )

    # --- 3. Set up the Trainer for the Search ---
    trainer = Trainer(
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # --- 4. Run Hyperparameter Search ---
    print("\n--- 🚀 Starting Hyperparameter Search... ---")
    best_run = trainer.hyperparameter_search(
        n_trials=20,
        direction="maximize",
        hp_space=lambda _ : {
            "learning_rate": _.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "num_train_epochs": _.suggest_int("num_train_epochs", 2, 4),
        }
    )

    print("--- Search Complete ---")
    print(f"Best Macro F1 (on validation split): {best_run.objective}")
    print(f"Best Hyperparameters: {best_run.hyperparameters}")

    # --- 5. Train the FINAL model on 100% of eng.csv ---
    print("\n--- 🚂 Training Final Model on 100% of 'eng.csv'... ---")

    # Update the trainer args with the best hyperparameters
    for k, v in best_run.hyperparameters.items():
        setattr(training_args, k, v)

    # --- ✅ CRITICAL FIX ---
    # The hyperparameter search can pollute the original 'training_args'
    # object and set evaluation_strategy to "no".
    # We must explicitly re-set it here to match our save_strategy.
    setattr(training_args, "evaluation_strategy", "epoch")

    # Create a new, final trainer
    final_trainer = Trainer(
        model=model_init(), # A new model
        args=training_args,
        train_dataset=tokenized_full_train, # Train on 100%
        eval_dataset=tokenized_eval, # Still use eval set for best model saving
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    final_trainer.train()
    print("--- Final Training Complete ---")

    # --- 6. Run Predictions on the Unlabeled Test Set ---
    print(f"\n--- 🔮 Running predictions on '{TEST_FILE_PATH}'... ---")

    # This will have 133 rows
    predictions = final_trainer.predict(tokenized_test)
    y_pred = np.argmax(predictions.predictions, axis=1)

    # --- 7. Save predictions in the correct format ---
    print(f"\n--- 💾 Saving predictions to '{OUTPUT_FILE_PATH}' ---")
    try:
        # Get the 'id's from the original test dataset
        ids = test_dataset[ID_COLUMN]

        # This DataFrame will have 133 rows
        predictions_df = pd.DataFrame({
            ID_COLUMN: ids,
            LABEL_COLUMN: y_pred
        })

        predictions_df.to_csv(OUTPUT_FILE_PATH, index=False)
        print("Successfully saved final predictions.")
        print(f"File created: {OUTPUT_FILE_PATH} (with {len(predictions_df)} rows)")
        print("\n🎉 You can now download this file and use it with your evaluation script! 🎉")

    except KeyError:
        print(f"Error saving predictions: '{ID_COLUMN}' column not found in '{TEST_FILE_PATH}'.")
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")

if __name__ == "__main__":
    main()
