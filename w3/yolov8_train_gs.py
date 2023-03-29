import itertools
from ultralytics import YOLO

def train_model(data, epochs, imgsz, batch, lr0, optimizer):
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr0, optimizer=optimizer)

# Define the hyperparameter space for the grid search
epochs_options = [50, 100]
imgsz_options = [640, 1024]
batch_size_options = [8, 16]
learning_rate_options = [0.001, 0.01]
optimizer_options = ['SGD', 'Adam', 'AdamW', 'RMSProp']

# Generate all combinations of hyperparameters
hyperparameter_space = list(itertools.product(epochs_options, imgsz_options, batch_size_options, learning_rate_options, optimizer_options))

# Perform grid search
for epochs, imgsz, batch, lr0, optimizer in hyperparameter_space:
    print(f"Training with epochs={epochs}, imgsz={imgsz}, batch_size={batch}, learning_rate={lr0}, optimizer={optimizer}")
    train_model(data='data.yaml', epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr0, optimizer=optimizer)
