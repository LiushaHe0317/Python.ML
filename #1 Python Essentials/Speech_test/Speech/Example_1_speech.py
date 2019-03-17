
import speech_recognition as sreg
from gtts import gTTS
from playsound import playsound

def recording_the_sound():
	r = sreg.Recognizer()

	with sreg.Microphone() as source:
    		audio = r.listen(source)

	text = r.recognize_google(audio,language='zh')

	tts = gTTS(text,lang = 'zh')
	tts.save('Repeat.mp3')
	playsound('Repeat.mp3')

	print(text)

if __name__ == '__main__':
	recording_the_sound()