import ollama
import sounddevice as sd
import soundfile as sf
import vosk
import json
import pyttsx3
import numpy as np
import sys
import threading
import webrtcvad

class VoiceAssistant:
    def __init__(self, sample_rate=16000, frame_duration=30):
        # Initialize speech recognition
        self.recognizer = vosk.Model("./vosk-model-en-us-0.22-lgraph")
        
        # Initialize text-to-speech
        self.speaker = pyttsx3.init()
        voices = self.speaker.getProperty('voices')
        self.speaker.setProperty('voice', voices[0].id)
        
        # Audio settings
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * (frame_duration / 1000))
        self.vad = webrtcvad.Vad(3)  # Aggressive mode
        
        # State management
        self.is_speaking = False
        self.stop_speaking = False

    def is_speech(self, frame):
        try:
            return self.vad.is_speech(frame.tobytes(), self.sample_rate)
        except:
            return False

    def safe_speak(self, text):
        self.is_speaking = True
        self.stop_speaking = False
        
        # Start interruption monitoring
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=self.monitor_for_interruption, args=(stop_event,))
        monitor_thread.start()
        
        try:
            # Speak the text
            self.speaker.say(text)
            self.speaker.runAndWait()
        except Exception as e:
            print(f"Speaking error: {e}")
        finally:
            stop_event.set()
            monitor_thread.join()
            self.is_speaking = False

    def monitor_for_interruption(self, stop_event):
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            blocksize=self.frame_size
        ) as stream:
            while not stop_event.is_set():
                frame, _ = stream.read(self.frame_size)
                if self.is_speech(frame):
                    print("\nInterruption detected! Stopping response...")
                    self.speaker.stop()
                    self.stop_speaking = True
                    break

    def listen(self):
        print("\nListening... (Speak now)")
        try:
            frames = []
            silence_counter = 0
            max_silence = 1.5  # seconds of silence to stop
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                blocksize=self.frame_size
            ) as stream:
                while True:
                    frame, _ = stream.read(self.frame_size)
                    frames.append(frame)
                    
                    if self.is_speech(frame):
                        silence_counter = 0
                    else:
                        silence_counter += self.frame_duration / 1000
                    
                    if silence_counter >= max_silence and len(frames) > 10:
                        break
                    
                    if self.stop_speaking:
                        return ""

            audio = np.concatenate(frames)
            sf.write('input.wav', audio, self.sample_rate)
            return self.transcribe_audio('input.wav')
        
        except Exception as e:
            print(f"Recording error: {e}")
            return ""

    def transcribe_audio(self, audio_file):
        try:
            recognizer = vosk.KaldiRecognizer(self.recognizer, self.sample_rate)
            with sf.SoundFile(audio_file) as f:
                while True:
                    data = f.read(4000, dtype='int16')
                    if len(data) == 0:
                        break
                    recognizer.AcceptWaveform(data.tobytes())
            return json.loads(recognizer.FinalResult())['text']
        except:
            return ""

    def generate_response(self, text):
        try:
            response = ollama.chat(model='deepseek-r1:8b', messages=[
                {'role': 'user', 'content': text}
            ])
            return response['message']['content']
        except Exception as e:
            print(f"API error: {e}")
            return "Sorry, I couldn't generate a response."

    def run(self):
        try:
            while True:
                user_input = self.listen()
                
                if self.stop_speaking:
                    self.stop_speaking = False
                    continue
                
                if not user_input:
                    continue
                    
                print(f"You: {user_input}")
                
                if user_input.lower() in ["exit", "stop", "quit"]:
                    print("Goodbye!")
                    break
                
                ai_response = self.generate_response(user_input)
                print(f"Assistant: {ai_response}")
                self.safe_speak(ai_response)
        
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

if __name__ == "__main__":
    print("Starting voice assistant...")
    assistant = VoiceAssistant()
    assistant.run()