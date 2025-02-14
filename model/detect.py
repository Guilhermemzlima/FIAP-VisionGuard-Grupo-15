# model/detect.py
import cv2
from ultralytics import YOLO
from alert.email_alert import send_email_alert

# Define the confidence threshold
CONFIDENCE_THRESHOLD = 0.5
# Define the classes that should trigger an alert
ALERT_CLASSES = [ 'knife', 'pistol' ]


def process_frame(frame, model):
    detections_made = False

    # Perform prediction with YOLOv8
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD)

    # Iterate over the results (usually just one image)
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # Get the coordinates, confidence, and classes of each detected box
            xyxy = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for (box, conf, cls) in zip(xyxy, confs, classes):
                if conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box
                    # Get the class name (using the model.model.names dictionary)
                    names = model.model.names if hasattr(model.model, 'names') else {}
                    name = names.get(int(cls), str(int(cls)))

                    # Draw the box and label on the image
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Check if the detected class is in the alert classes
                    if name in ALERT_CLASSES:
                        detections_made = True
    return frame, detections_made


def main():
    # Caminho para o modelo treinado customizado (ajuste conforme necessário)
    # model_path = "../yolov8/runs/detect/train/weights/best.pt"
    model_path = "./yolov8m.pt"
    model = YOLO(model_path)

    cap = cv2.VideoCapture(
        '/F:\Faculdade/video3.mp4')  # Abre a webcam. Para usar um vídeo, substitua o 0 pelo caminho do arquivo.

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, alert_triggered = process_frame(frame, model)

        if alert_triggered:
            # Envia alerta via e-mail (implemente lógica para evitar envios repetidos se necessário)
            send_email_alert(processed_frame)

        cv2.imshow("YOLOv8 - Detecção de Objetos Cortantes", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
