# README for Pegasus Summarization Project

## Project Overview

This project implements a text summarization model using the Pegasus architecture from Hugging Face's Transformers library. The model is trained on the SAMSum dataset, which consists of dialogues and their corresponding summaries. The goal is to generate concise summaries from conversational text.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [License](#license)

## Requirements

- Python 3.10 or higher
- PyTorch
- Transformers
- Datasets
- Evaluate
- NLTK
- Matplotlib

## Installation

To install the required packages, run the following commands:

```bash
pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr -q
pip install --upgrade accelerate
pip install evaluate
```

## Dataset

The project uses the SAMSum dataset, which contains dialogues and their summaries. The dataset is loaded using the `datasets` library:

```python
from datasets import load_dataset
dataset_samsum = load_dataset("samsum")
```

## Model Training

The Pegasus model is fine-tuned on the SAMSum dataset. The training process involves the following steps:

1. **Data Preparation**: Convert dialogues and summaries into input features.
2. **Training**: Use the `Trainer` class from the Transformers library to train the model.

### Example Code

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

# Define training arguments
trainer_args = TrainingArguments(
    output_dir='pegasus-samsum', 
    num_train_epochs=1, 
    per_device_train_batch_size=1,
    evaluation_strategy='steps',
    eval_steps=500,
)

trainer = Trainer(
    model=model_pegasus, 
    args=trainer_args,
    train_dataset=dataset_samsum_pt["train"],
    eval_dataset=dataset_samsum_pt["validation"]
)

trainer.train()
```

## Evaluation

The model's performance is evaluated using the ROUGE metric. The evaluation function calculates the ROUGE scores for the generated summaries against the reference summaries.

### Example Code

```python
import evaluate

rouge_metric = evaluate.load('rouge')

def calculate_metric_on_test_ds(test_dataset, metric, model, tokenizer):
    # Evaluation logic here
    return result

score = calculate_metric_on_test_ds(dataset_samsum['test'], rouge_metric, model_pegasus, tokenizer)
```

## Usage

After training, the model can be used to generate summaries for new dialogues. The following code demonstrates how to use the trained model for inference:

```python
from transformers import pipeline

pipe = pipeline("summarization", model="pegasus-samsum-model", tokenizer=tokenizer)
sample_text = dataset_samsum["test"][0]["dialogue"]
summary = pipe(sample_text, max_length=128, num_beams=8, length_penalty=0.8)[0]["summary_text"]

print("Model Summary:")
print(summary)
```

## Results

The model achieved the following ROUGE scores on the test set:

| Metric   | Score     |
|----------|-----------|
| ROUGE-1  | 0.328899  |
| ROUGE-2  | 0.092474  |
| ROUGE-L  | 0.255343  |
| ROUGE-Lsum | 0.258356 |

## Saving and Loading the Model

To save the trained model and tokenizer, use the following commands:

```python
model_pegasus.save_pretrained("pegasus-samsum-model")
tokenizer.save_pretrained("tokenizer")
```

To load the model and tokenizer later:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("pegasus-samsum-model")
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained("pegasus-samsum-model")
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive overview of the Pegasus Summarization Project, including setup, training, evaluation, and usage instructions.
