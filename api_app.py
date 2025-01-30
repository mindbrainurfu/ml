from fastapi import FastAPI, UploadFile, File, Response
# from transformers import pipeline
from faster_whisper import WhisperModel
import os
import torch
import librosa
from pathlib import Path
from api_mistral import load_model, generate_answer
from api_fish import process_and_run_commands
from pydub import AudioSegment
from api_emo import predict_emotion

app = FastAPI()

model_size = "medium"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(model_size, device=device)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_location = f"./temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
        
    try:
        segments, _ = model.transcribe(file_location)
        transcription = ' '.join([segment.text for segment in segments])
        return {"filename": file.filename, "transcription": transcription}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

@app.post("/get_emotion/")
async def get_emotion(file: UploadFile = File(...)):
    file_location = f"./temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    audio_path = Path(file_location)
    audio, sr = librosa.load(audio_path, sr=16000)  
    emotion = predict_emotion(audio, sr)
    return {'client_emotion': emotion}

mistral_model, tokenizer, device, system_prompt = load_model()

@app.get("/get_answer_mistral/")
async def mistral_get_answer(question: str):
    answer = generate_answer(question, mistral_model, tokenizer, device, system_prompt)
    return {"answer": answer}

@app.get("/synthesize_answer/")
async def fish_synthesize_answer(text: str):
    synthesized_answer = process_and_run_commands(text)
    audio = AudioSegment.from_file(synthesized_answer)
    audio_bytes = audio.export(format="wav").read()

    return Response(content=audio_bytes, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
