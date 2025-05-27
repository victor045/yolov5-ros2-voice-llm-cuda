# YOLOv5 ROS2 Voice Assistant with LLM

This project combines **YOLOv5 object detection**, **ROS2**, and a **voice-based AI assistant** powered by OpenAI's GPT-4 and Google's Speech Recognition API. It enables a robot or camera system to:

- Detect objects in real-time using YOLOv5
- Listen to user voice questions
- Understand whether the question is related to what it "sees"
- Answer the user via voice using LLM-generated responses

---

## 📸 Features

- 🔍 Real-time object detection via YOLOv5n (ultralytics)
- 🎙️ Voice input using microphone (speech recognition)
- 💬 Language understanding via OpenAI GPT-4
- 🗣️ Text-to-speech responses with `gTTS`
- 🧠 Relevance filtering to only respond to valid visual queries
- 📝 Interaction logging to `/tmp/voice_interactions.txt`

---

## 🚀 Requirements

- Python 3.8+
- ROS2 (Foxy or later)
- Torch + torchvision
- OpenAI Python SDK (`openai`)
- gTTS, mpg123
- cv_bridge, OpenCV, NumPy
- bboxes_ex_msgs.msg and sensor_msgs (ROS2 messages)

Install dependencies:
```bash
pip install -r requirements.txt
sudo apt install mpg123 portaudio19-dev
```

---

## 🔧 Setup

### 1. Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/yolov5-ros2-voice-llm.git
cd yolov5-ros2-voice-llm
```

### 2. Set your OpenAI API Key
Export your key in your shell session:
```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Run the ROS2 node
Assuming your ROS2 workspace is built:
```bash
ros2 run v4l2_camera v4l2_camera_node
```
in another shell:
```bash
ros2 run yolov5_ros yolov5_node
```

---

## 🎤 Usage Example

1. A person steps in front of the camera
2. YOLOv5 detects objects (e.g., "person")
3. The assistant says: "Estoy escuchando..."
4. You ask: _"¿Qué es lo que ves?"_
5. The system replies: _"Estoy viendo una persona y una silla."_

---

## 📁 File Structure

```bash
scripts/
├── yolov5_node.py          # Main node: detection + LLM response
config/
├── yolov5n.pt              # (Optional: model weights if using local copy)
data/
├── test_images/            # (Optional: samples)
logs/
├── voice_interactions.txt  # Interaction history
```

---

## 📚 Credits
- YOLOv5 by [Ultralytics](https://github.com/ultralytics/yolov5)
- SpeechRecognition and PyAudio
- GPT-4 via OpenAI
- ROS2 + rclpy

---

## 📜 License
MIT License
