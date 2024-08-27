#pre-train dataset and then fine-tune and make prediction
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained LLM
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Define the hyperparameters
learning_rate = 5e-5
epochs = 5

# Load the training data
train_data = torch.load("train_data.pt")


# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    for batch in train_data:
        # Forward pass
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

        # Calculate the loss
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
torch.save(model.state_dict(), "fine-tuned_model.pt")


# Load the fine-tuned model
model = torch.load("fine-tuned_model.pt")

# Classify a text sequence
text_sequence = "This is a positive review."
outputs = model(tokenizer(text_sequence, return_tensors="pt")["input_ids"])

# Get the predicted label
predicted_label = outputs.logits.argmax(-1)

# Print the predicted label
print(predicted_label)