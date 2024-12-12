import torch
from src.train import model
from src.datasets.accelerometry import window_size, num_features


def main():
    # Set up testing data
    num_samples_test = 200
    X_test = torch.randn(num_samples_test, window_size, num_features)
    X_test = X_test.to("mps")
    print("X_test shape:", X_test.shape)

    # Set model to evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(X_test)

    # Extract predicted classes
    _, predicted_classes = torch.max(outputs, 1)
    print("Predicted classes:", predicted_classes, "\nShape:", predicted_classes.shape)


if __name__ == "__main__":
    main()
