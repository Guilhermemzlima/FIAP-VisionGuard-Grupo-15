# alert/email_alert.py
import smtplib
from email.message import EmailMessage
import cv2
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def send_email_alert(frame):
    # Save the frame to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_path = tmp.name
    cv2.imwrite(image_path, frame)

    # Email configuration
    msg = EmailMessage()
    msg['Subject'] = "Alerta: Objeto Cortante Detectado"
    msg['From'] = os.getenv("SMTP_USER")  # Update to use environment variable
    msg['To'] = os.getenv("SMTP_SEND")  # Update to the security center email
    msg.set_content("Foi detectado um objeto cortante pela câmera de segurança. Verifique imediatamente.")

    # Attach the image
    with open(image_path, 'rb') as f:
        file_data = f.read()
    msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename='alert.jpg')

    # SMTP server configuration (example with Gmail)
    smtp_server = "smtp.gmail.com"
    smtp_port = 465
    smtp_user = os.getenv("SMTP_USER")  # Update to use environment variable
    smtp_password = os.getenv("SMTP_PASSWORD")  # Update to use environment variable

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)
        print("Alerta enviado por e-mail.")
    except Exception as e:
        print("Falha ao enviar e-mail:", e)
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
