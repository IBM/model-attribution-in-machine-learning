from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
dataset = load_dataset('imdb')



model = AutoModelForSequenceClassification.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
training_args = TrainingArguments(output_dir="./mlmini_imdb", evaluation_strategy="epoch",
                                  num_train_epochs=10, save_strategy='epoch', )

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'].shuffle(seed=42).select(range(10000)),
    eval_dataset=tokenized_datasets['test'].shuffle(seed=42).select(range(1000)),
    compute_metrics=compute_metrics,
)

trainer.train()