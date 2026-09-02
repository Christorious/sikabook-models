from vosk import Model, KaldiRecognizer
import wave
import json

MODEL_PATH = "models/sikabook-en-gh-v1"
WAV_FILE = "/home/christian/sikabook-models/kaldi/src/feat/test_data/test.wav"

wf = wave.open(WAV_FILE, "rb")

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, wf.getframerate())

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break

    rec.AcceptWaveform(data)

print(json.loads(rec.FinalResult()))
