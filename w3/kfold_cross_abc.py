import numpy as np
from sklearn.model_selection import KFold, train_test_split
from ultralytics import YOLO


def train_and_evaluate_model(train_idx, val_idx, fold_number, epochs=100, imgsz=640):
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    with open('data.yaml', 'r') as f:
        data_yaml = f.read()

    train_data = [f"train_images/{i}.jpg" for i in train_idx]
    val_data = [f"val_images/{i}.jpg" for i in val_idx]

    data_yaml = data_yaml.replace("train: train_images/*", f"train: {','.join(train_data)}")
    data_yaml = data_yaml.replace("val: val_images/*", f"val: {','.join(val_data)}")

    with open(f'data_fold{fold_number}.yaml', 'w') as f:
        f.write(data_yaml)

    model.train(data=f'data_fold{fold_number}.yaml', epochs=epochs, imgsz=imgsz)

    return model

def strategy_a(X, epochs=100, imgsz=640):
    train_idx, test_idx = X[:len(X) // 4], X[len(X) // 4:]
    print("Training on Strategy A")
    model = train_and_evaluate_model(train_idx, test_idx, 'A', epochs, imgsz)

def strategy_b(X, k=4, epochs=100, imgsz=640):
    kf = KFold(n_splits=k)
    fold_number = 1

    for train_idx, val_idx in kf.split(X):
        print(f"Training on Strategy B - Fold {fold_number}")
        model = train_and_evaluate_model(train_idx, val_idx, f'B_{fold_number}', epochs, imgsz)
        fold_number += 1

def strategy_c(X, k=4, epochs=100, imgsz=640):
    for i in range(1, k+1):
        train_idx, val_idx = train_test_split(X, train_size=0.25, random_state=i)
        print(f"Training on Strategy C - Iteration {i}")
        model = train_and_evaluate_model(train_idx, val_idx, f'C_{i}', epochs, imgsz)
"""
if __name__ == "__main__":
    X = np.arange(255)

    strategy_a(X)
    strategy_b(X)
    strategy_c(X)
"""