# alert/email_alert.py
import smtplib
from email.message import EmailMessage
import cv2
import tempfile
import os


def send_email_alert(frame):
    # Salva o frame em um arquivo temporário
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_path = tmp.name
    cv2.imwrite(image_path, frame)

    # Configuração do e-mail
    msg = EmailMessage()
    msg['Subject'] = "Alerta: Objeto Cortante Detectado"
    msg['From'] = "starkgamerbr.bazinga@gmail.com"  # Atualize para o seu e-mail
    msg['To'] = "guilherme.munhozlima@gmail.com"  # Atualize para o e-mail da central de segurança
    msg.set_content("Foi detectado um objeto cortante pela câmera de segurança. Verifique imediatamente.")

    # Anexa a imagem
    with open(image_path, 'rb') as f:
        file_data = f.read()
    msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename='alert.jpg')

    # Configuração do servidor SMTP (exemplo com Gmail)
    smtp_server = "smtp.gmail.com"
    smtp_port = 465
    smtp_user = "starkgamerbr.bazinga@gmail.com"  # Atualize
    smtp_password = "ofhp abcj iqyu qwfk"  # Atualize (considere usar variáveis de ambiente ou outro método seguro)

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
