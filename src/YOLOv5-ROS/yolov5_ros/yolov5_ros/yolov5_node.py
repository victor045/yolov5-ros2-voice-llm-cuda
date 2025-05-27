import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np

from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox

from openai import OpenAI, OpenAIError
import os
import threading
import ctypes
import subprocess
from gtts import gTTS
import uuid
import traceback
import time
import speech_recognition as sr

MIC_FILE = "/tmp/working_mic.txt"
HISTORY_FILE = "/tmp/voice_interactions.txt"
listen_lock = threading.Lock()
last_voice_interaction = 0
COOLDOWN_SECONDS = 10

# Ocultar errores molestos de ALSA
try:
    asound = ctypes.CDLL('libasound.so')
    asound.snd_lib_error_set_handler(None)
except Exception:
    pass

client = OpenAI()

def transcribe_with_speech_recognition():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("🎙️ Ajustando al ruido de ambiente...")
            recognizer.adjust_for_ambient_noise(source)
            print("🎤 Grabando voz...")
            audio = recognizer.listen(source, timeout=5)
            print("🧠 Transcribiendo...")
            text = recognizer.recognize_google(audio, language="es-MX")
            print("✅ Se reconoció:", text)
            return text
    except sr.UnknownValueError:
        print("⚠️ No se entendió el audio.")
        return ""
    except sr.WaitTimeoutError:
        print("⏱️ Tiempo agotado sin voz.")
        return ""
    except sr.RequestError as e:
        print(f"❌ Error con el servicio de reconocimiento: {e}")
        return ""
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return ""

def es_pregunta_relevante_llm(question):
    # Aquí devolvemos siempre True para permitir que todas las preguntas sean procesadas
    return True

def listen_and_respond(detected_objects):
    global last_voice_interaction
    if not listen_lock.acquire(blocking=False):
        print("🔁 Ya hay una interacción de voz en curso.")
        return

    now = time.time()
    if now - last_voice_interaction < COOLDOWN_SECONDS:
        print("⏳ En espera por cooldown.")
        listen_lock.release()
        return

    try:
        question = transcribe_with_speech_recognition()

        if not question:
            print("❌ No se pudo transcribir.")
            return

        print(f"🧠 Pregunta: {question}")

        if not es_pregunta_relevante_llm(question):
            print("⚠️ El LLM dice que la pregunta no es relevante.")
            return

        prompt = f"El robot detectó los siguientes objetos: {detected_objects}. Responde esto: '{question}'"

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        print("🤖 Respuesta:", answer)
        speak(answer)
        last_voice_interaction = time.time()

        with open(HISTORY_FILE, "a") as f:
            f.write(f"Pregunta: {question}\nRespuesta: {answer}\n---\n")

    except Exception as e:
        print(f"❌ Error general en listen_and_respond:")
        traceback.print_exc()
    finally:
        listen_lock.release()

def speak(text):
    try:
        tts = gTTS(text=text, lang='es')
        filename = f"/tmp/{uuid.uuid4()}.mp3"
        tts.save(filename)
        subprocess.run(["mpg123", filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(filename)
    except Exception as e:
        print(f"❌ Error con gTTS: {e}")

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            self.get_logger().info("Running on CUDA")
        else:
            self.get_logger().info("Running on CPU")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        self.publisher = self.create_publisher(BoundingBoxes, '/bounding_boxes', 10)
        self.image_pub = self.create_publisher(Image, '/yolo/image_result', 10)

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame)

        bbox_msg = BoundingBoxes()
        bbox_msg.header = msg.header

        detected_objects = []

        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            bbox = BoundingBox()
            bbox.xmin = int(xyxy[0])
            bbox.ymin = int(xyxy[1])
            bbox.xmax = int(xyxy[2])
            bbox.ymax = int(xyxy[3])
            bbox.class_id = str(int(cls))
            bbox_msg.bounding_boxes.append(bbox)

            label = self.model.names[int(cls)]
            detected_objects.append(label)

            cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("YOLOv5 Detection", frame)
        cv2.waitKey(1)

        self.publisher.publish(bbox_msg)

        annotated_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        annotated_img_msg.header = msg.header
        self.image_pub.publish(annotated_img_msg)

        if detected_objects:
            threading.Thread(target=listen_and_respond, args=(detected_objects,), daemon=True).start()

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("🛑 Interrupción del usuario (Ctrl+C)")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
