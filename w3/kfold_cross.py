import numpy as np
from sklearn.model_selection import KFold
from ultralytics import YOLO

def train_model(train_idx, val_idx, fold_number, epochs=100, imgsz=640):
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Modify data.yaml to have the correct train and validation set for this fold
    with open('data.yaml', 'r') as f:
        data_yaml = f.read()

    train_data = [f"train_images/{i}.jpg" for i in train_idx]
    val_data = [f"val_images/{i}.jpg" for i in val_idx]

    data_yaml = data_yaml.replace("train: train_images/*", f"train: {','.join(train_data)}")
    data_yaml = data_yaml.replace("val: val_images/*", f"val: {','.join(val_data)}")

    with open(f'data_fold{fold_number}.yaml', 'w') as f:
        f.write(data_yaml)

    # Train the model
    model.train(data=f'data_fold{fold_number}.yaml', epochs=epochs, imgsz=imgsz)

    return model

def k_fold_cross_validation(k, X, epochs=100, imgsz=640):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_number = 1

    for train_idx, val_idx in kf.split(X):
        print(f"Training on fold {fold_number}")
        model = train_model(train_idx, val_idx, fold_number, epochs, imgsz)
        model.save(f'yolov8n_fold{fold_number}.pt')
        fold_number += 1

if __name__ == "__main__":
    k = 4  # Number of folds
    X = np.arange(255) 

    k_fold_cross_validation(k, X)
