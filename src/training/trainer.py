import torch
import torch.nn as nn
import torch.optim as optim
import time

from src.config import NUM_EPOCHS, LEARNING_RATE, DEVICE


def calculate_accuracy(model, val_loader, verbose=True):
    model = model.to(DEVICE)
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if verbose:
                print(f"correct: {correct}, total: {total}")

    accuracy = correct / total
    if verbose:
        print(f"Validation Accuracy: {100 * accuracy:.2f}%")
    return accuracy


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
):
    print("Moving model to device...")
    model = model.to(DEVICE)

    print("Defining loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    accuracy_list = []

    start_time = time.time()

    print("Training loop...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        print("Evaluating model...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        accuracy = calculate_accuracy(model, val_loader, verbose=False)
        accuracy_list.append(accuracy)

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {100 * accuracy:.2f}%"
        )

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    return train_losses, val_losses, accuracy_list
