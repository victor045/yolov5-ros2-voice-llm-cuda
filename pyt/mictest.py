import speech_recognition as sr

def listen_from_microphone():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("🎙️ Adjusting for ambient noise...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎤 Listening now! Speak something.")
        try:
            audio = recognizer.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected (timeout).")
            return ""

    print("⏳ Recognizing...")
    try:
        text = recognizer.recognize_google(audio)
        print("✅ You said:", text)
        return text
    except sr.UnknownValueError:
        print("⚠️ Could not understand the audio.")
    except sr.RequestError as e:
        print(f"❌ Could not request results from Google: {e}")
    return ""

listen_from_microphone()

