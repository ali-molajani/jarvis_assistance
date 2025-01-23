import ollama
import sounddevice as sd
import soundfile as sf
import vosk
import json
import pyttsx3
import numpy as np
import webrtcvad

class VoiceAssistant:
    def __init__(self, sample_rate=16000, frame_duration=30):
        # Initialize speech recognition
        self.recognizer = vosk.Model("./vosk-model-en-us-0.22-lgraph")
        
        # Initialize text-to-speech
        self.speaker = pyttsx3.init()
        voices = self.speaker.getProperty('voices')
        self.speaker.setProperty('voice', voices[0].id)
        
        # VAD setup
        self.vad = webrtcvad.Vad(3)  # Aggressive mode
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * (frame_duration / 1000))
    
    def is_speech(self, frame):
        """Determine if the frame contains speech."""
        try:
            return self.vad.is_speech(frame.tobytes(), self.sample_rate)
        except:
            return False
    
    def listen(self, max_silence_duration=1.0):
        print("Listening...")
        try:
            # Record with dynamic stopping
            frames = []
            silence_counter = 0
            max_silence_frames = int(max_silence_duration * 1000 / self.frame_duration)
            
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16') as stream:
                while True:
                    frame, _ = stream.read(self.frame_size)
                    
                    if self.is_speech(frame):
                        frames.append(frame)
                        silence_counter = 0
                    else:
                        silence_counter += 1
                    
                    # Stop if prolonged silence detected
                    if silence_counter > max_silence_frames and frames:
                        break
            
            # Convert frames to numpy array
            audio = np.concatenate(frames, axis=0)
            
            # Save audio
            sf.write('input.wav', audio, self.sample_rate)
            return self.transcribe_audio('input.wav')
        
        except Exception as e:
            print(f"Error during recording: {e}")
            return ""
    
    # Rest of the methods remain the same as in your previous implementation
    def transcribe_audio(self, audio_file):
        recognizer = vosk.KaldiRecognizer(self.recognizer, self.sample_rate)
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