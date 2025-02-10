# model/train.py
import torch
import os
from ultralytics import YOLO


def train_yolov8():
    # Caminho absoluto para o arquivo data.yaml
    data_config = os.path.abspath(os.path.join(".", "configs", "data.yaml"))

    # Escolha do modelo pré-treinado; aqui usamos a variante "nano"
    # Outras opções: yolov8s.pt, yolov8m.pt, yolov8l.pt, etc.
    model = YOLO("yolov8n.pt")

    # Inicia o treinamento
    model.train(
        data=data_config,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        half=True,
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()  # Clear GPU cache
    train_yolov8()
