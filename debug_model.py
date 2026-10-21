import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("Artifacts/BiGRU.keras", compile=False)
with open("Artifacts/tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

maxlen = model.input_shape[1] or 100
print("model input shape :", model.input_shape)
print("model output shape:", model.output_shape)
print("tokenizer vocab   :", len(tok.word_index), "words, oov_token =", tok.oov_token)

print("\nVocabulary check")
for word in ["the", "is", "i", "angry", "angri", "happy", "happi", "scared", "scare"]:
    print(f"  {word:8} in vocab: {word in tok.word_index}")

tests = {
    "joy":      "i am feeling very happy today",
    "sadness":  "i feel so sad and lonely and i want to cry",
    "anger":    "this situation is really making me angry",
    "fear":     "i am so scared about what might happen",
    "love":     "i love you so much and i care deeply about you",
    "surprise": "i was shocked and amazed by the sudden news",
}

for padding in ("post", "pre"):
    print(f"\npadding = {padding}")
    for expected, text in tests.items():
        x = pad_sequences(tok.texts_to_sequences([text]), maxlen=maxlen, padding=padding)
        p = model.predict(x, verbose=0)[0]
        print(f"  {expected:9} -> index {p.argmax()}  ({p.max():.0%})  {np.round(p, 2)}")
