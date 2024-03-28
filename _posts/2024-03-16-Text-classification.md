---
layout: post
title: Text classification using pre trained Albert model from Hugging Face
date: 2024-03-15 13:32:20 +0300
description: Using machine learning model on a dataset from an American airline, this project predicts passenger satisfaction with airline services, highlighting key factors influencing satisfaction and provides actionable recommendations for enhancing customer experience across different flight categories and passenger classes. # Add post description (optional)
img: text.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Text classification, Sentiment Analysis, AlbertV5, Huggingface]
---
ALBERT is a transformer-based language representation model developed by Google. ALBERT aims to provide a more efficient and compact architecture compared to BERT while maintaining competitive performance on various NLP tasks.

In this project, I am using amazon review dataset to perform sentiment analysis using a pre-trained ALBERT model sourced from the Hugging Face library. 

Data Source: url = ["https://s3.amazonaws.com/fast-ai-nlp/amazon_review_full_csv.tgz"]("https://s3.amazonaws.com/fast-ai-nlp/amazon_review_full_csv.tgz")

### Installing dependencies

First, I have installed necessary dependencies using 'pip'. The 'transformers' library and 'sentencepiece' are installed. 


```python
pip install -q transformers
```

SentencePiece is an unsupervised text tokenizer and detokenizer which allows the vocabulary size to be predetermined before training the neural model.


```python
pip install sentencepiece
```

The next block of code sets up the device for GPU usage.


```python
# # Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
device
```
## Importing Libraries
```python
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers
import requests
from io import BytesIO
import tarfile


from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW
from transformers import AlbertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```

## Load data from source


```python
url = "https://s3.amazonaws.com/fast-ai-nlp/amazon_review_full_csv.tgz"
response = requests.get(url)
file = tarfile.open(fileobj=BytesIO(response.content))
file.extractall()
```

The training and test datasets are loaded into Pandas DataFrames (train and test). I have only included relevant columns 'Rating', 'Title', 'ReviewText' from the dataset.


```python
train = pd.read_csv('amazon_review_full_csv/train.csv',usecols=[0,1,2], names=['Rating', 'Title','ReviewText'],nrows=8000)
test = pd.read_csv('amazon_review_full_csv/test.csv',usecols=[0,1,2], names=['Rating', 'Title','ReviewText'],nrows=2500)
```


I have defined a 'classify_rating()' function to categorize ratings into different sentiment groups based on their numerical values.
If the rating is greater than 3, it is classified as 'Positive'. If the rating is less than 3, it is classified as 'Negative' else it is classified as 'Neutral'. Then, I have added new column 'Sentiment' based on the rating.


```python
# Define function to classify ratings
def classify_rating(rating):
    if rating > 3:
        return 'Positive'
    elif rating < 3:
        return 'Negative'
    else:
        return 'Neutral'
```


```python
train['Sentiment'] = train['Rating'].apply(lambda x: classify_rating(x))
test['Sentiment'] = test['Rating'].apply(lambda x: classify_rating(x))
```


```python
train,test = train[['Sentiment','ReviewText']], test[['Sentiment','ReviewText']]
```

## Convert labels to numeric values

The label2id dictionary is created to map sentiment labels to numerical IDs. And for further processing, I am assigning the numerical IDs back to the "Sentiment" column. 


```python

label2id = {"Positive": 2, "Neutral": 1, "Negative": 0}
train["label"] = train["Sentiment"].map(label2id)
test["label"] = test["Sentiment"].map(label2id)
```


```python
train["Sentiment"] = train["label"]
test["Sentiment"] = test["label"]
```

For model training, I am discarding other columns and just selecting 'Sentiment' and 'ReviewText' columns.


```python
train,test = train[['Sentiment','ReviewText']], test[['Sentiment','ReviewText']]
```

After preprocessing, the train and test dataframe looks as below.
![Getting Started](/assets/img/Text/9.png)


## Load the Albert tokenizer and encode the text:

The following code snippet instantiates an  AlbertTokenizerFast from the Hugging Face transformers library. This tokenizer is initialized with the pre-trained weights of the 'albert-base-v2' model. Then we have defined he 'AmazonReviewDataset' class to prepare the input data for the model.


```python
tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')

```


```python
class AmazonReviewDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.Sentiment = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.Sentiment[idx]
        encoding = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }
```
Each dataset contains review texts tokenized and encoded into input IDs and attention masks, along with their corresponding sentiment labels as PyTorch tensors.


```python
train_dataset = AmazonReviewDataset(train["ReviewText"], train["Sentiment"])
val_dataset = AmazonReviewDataset(test["ReviewText"], test["Sentiment"])
```

## Create data loaders:

I have used a batch size of 16 to reduce memory usage and improve computational efficiency during training and validation. The train_loader is created for the training dataset (train_dataset) which batches the data with a batch size of 16 and shuffles the data during each epoch, which helps in preventing the model from learning the order of the samples and improves generalization. The 'val_loader' is created for validation dataset for evaluating the model's performance.


```python
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

```

## Load the pre-trained Albert model for sequence classification:

I have initialized the ALBERT model for sequence classification(AlbertForSequenceClassification) from the pre-trained albert-base-v2 checkpoint. The model is moved to the appropriate device (GPU or CPU).

```python
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=len(label2id))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

## Define the optimizer and learning rate scheduler:

In the following code block, I have set up an Adam optimizer with a learning rate of 2e-5 and a learning rate scheduler that decreases the learning rate by 10% after each epoch.

```python
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 5  # 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
```
## Train the model:

I have set up the number of epochs to 5 in order to train the neural network. 

```python
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predictions = torch.max(logits, dim=1)
            val_predictions.extend(predictions.detach().cpu().numpy())
            val_labels.extend(labels.detach().cpu().numpy())
    val_acc = accuracy_score(val_labels, val_predictions)
    print(f"Validation Accuracy: {val_acc:.4f}")

```

    Epoch 1/5 - Avg Loss: 0.9975
    Validation Accuracy: 0.6308
    Epoch 2/5 - Avg Loss: 0.9900
    Validation Accuracy: 0.6308
    Epoch 3/5 - Avg Loss: 0.9900
    Validation Accuracy: 0.6308
    Epoch 4/5 - Avg Loss: 0.9908
    Validation Accuracy: 0.6308
    Epoch 5/5 - Avg Loss: 0.9906
    Validation Accuracy: 0.6308

It appears that the training loss is fluctuating slightly over epochs, while the validation accuracy remains constant at 63.08%. This could suggest that the model may not be learning effectively or that the dataset or model architecture may need further optimization or investigation

## Evaluate the model on training set

```python
# Evaluation on the train set

model.eval()
train_predictions = []
train_labels = []
with torch.no_grad():
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predictions = torch.max(logits, dim=1)
        train_predictions.extend(predictions.detach().cpu().numpy())
        train_labels.extend(labels.detach().cpu().numpy())

train_acc = accuracy_score(train_labels, train_predictions)
classification_rep = classification_report(train_labels, train_predictions, target_names=label2id.keys())

print(f"Train Accuracy: {train_acc:.4f}")
print("\nClassification Report:")
print(classification_rep)
```

    Train Accuracy: 0.6266
    
    Classification Report:
                  precision    recall  f1-score   support
    
        Positive       0.58      0.87      0.70      3236
         Neutral       0.00      0.00      0.00      1613
        Negative       0.69      0.69      0.69      3151
    
        accuracy                           0.63      8000
       macro avg       0.43      0.52      0.46      8000
    weighted avg       0.51      0.63      0.56      8000
    
When evaluated on the training set, the overall accuracy of the model on the training set is 62.66%. This indicates that the model correctly predicts the sentiment for approximately 62.66% of the reviews in the training dataset.

### Conclusion

This project leverages the power of ALBERT, a cutting-edge transformer-based language model, to tackle the task of sentiment analysis on Amazon reviews. Despite observing fluctuations in training loss, the model's validation accuracy remains stagnant at 63.08%, prompting further investigation into potential optimization method. Fine-tuning model hyperparameters, such as learning rate or batch size, could potentially enhance model learning and generalization.

