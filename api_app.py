from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import os
import torch

app = FastAPI()

model_name = "openai/whisper-medium"  
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    #сохранение файла
    file_location = f"./temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    #транскрипция аудио
    try:
        transcription = pipe(file_location)["text"]
        return {"filename": file.filename, "transcription": transcription}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

