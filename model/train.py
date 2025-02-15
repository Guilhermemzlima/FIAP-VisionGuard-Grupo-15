# model/train.py
import torch
import os
from ultralytics import YOLO


def train_yolo():
    # Caminho absoluto para o arquivo data.yaml
    data_config = os.path.abspath(os.path.join(".", "configs", "data.yaml"))

    # Escolha do modelo pré-treinado; aqui usamos a variante "nano"
    # Outras opções: yolov8s.pt, yolov8m.pt, yolov8l.pt, etc.
    model = YOLO("./yolo11l.pt")

    # Inicia o treinamento
    model.train(
        data=data_config,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()  # Clear GPU cache
    train_yolo()
