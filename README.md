# Offline assistance
I,m trying to make a assistance like what we see in the ironman to automate stuff icluding runing build scripts.
# Tools
Yes i know better engines out there specially for TTS models there are very powerful and more accurate models out there. now we are trying to just implemnt the basics then we are going to have different branches to get more features. (It's gonna be a very long Project)

**Tools**: we are using `vosk lgraph model` for speech to text, using ollama and `phi3.5` to generate the answers, and `pytts3` to text to speech conversion.

## TODO:
- [ ] replace the `pytts3` with [kokoro](https://huggingface.co/hexgrad/Kokoro-82M)
- [ ] use higher accuracy `vosk` model [vosk models](https://alphacephei.com/vosk/models) use main and past version
- [X] replace the engine from `phi3.5` to `deepseek` [ollama deepseek](https://ollama.com/library/deepseek-r1) even `deepseek coder`
