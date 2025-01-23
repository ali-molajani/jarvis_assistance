import ollama
import sounddevice as sd
import soundfile as sf
import vosk
import json
import pyttsx3

class VoiceAssistant:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = vosk.Model("./vosk-model-en-us-0.22-lgraph")
        
        # Initialize text-to-speech
        self.speaker = pyttsx3.init()
        voices = self.speaker.getProperty('voices')
        self.speaker.setProperty('voice', voices[0].id)  # Set voice
    
    def listen(self):
        print("Listening...")
        try:
            audio = sd.rec(int(3 * 44100), samplerate=44100, channels=1, dtype='int16')
            sd.wait()
            sf.write('input.wav', audio, 44100)
            return self.transcribe_audio('input.wav')
        except Exception as e:
            print(f"Error during recording: {e}")
            return ""
    
    def transcribe_audio(self, audio_file):
        recognizer = vosk.KaldiRecognizer(self.recognizer, 44100)
        with sf.SoundFile(audio_file) as audio:
            while True:
                data = audio.read(4000, dtype='int16')
                if len(data) == 0:
                    break
                recognizer.AcceptWaveform(data.tobytes())
            result = json.loads(recognizer.FinalResult())
            return result.get('text', '')
    
    def generate_response(self, text):
        try:
            response = ollama.chat(model='phi3.5', messages=[
                {'role': 'user', 'content': text}
            ])
            return response['message']['content']
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response."
    
    def speak(self, text):
        self.speaker.say(text)
        self.speaker.runAndWait()
    
    def run(self):
        while True:
            user_input = self.listen()
            if user_input.lower() in ["exit", "stop", "quit"]:
                print("Goodbye!")
                break
            ai_response = self.generate_response(user_input)
            self.speak(ai_response)

# Initialize and run assistant
assistant = VoiceAssistant()
assistant.run()