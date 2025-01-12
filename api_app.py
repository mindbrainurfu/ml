from fastapi import FastAPI, UploadFile, File
# from transformers import pipeline
from faster_whisper import WhisperModel
import os
import torch

app = FastAPI()

# model_name = "openai/whisper-medium"  
model_size = "medium"
device = "cuda" if torch.cuda.is_available() else "cpu"
# pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)
model = WhisperModel(model_size, device=device)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    #сохранение файла
    file_location = f"./temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    #транскрипция аудио
    # try:
    #     transcription = pipe(file_location)["text"]
    #     return {"filename": file.filename, "transcription": transcription}
    # except Exception as e:
    #     return {"error": str(e)}
    # finally:
    #     if os.path.exists(file_location):
    #         os.remove(file_location)
    try:
        segments, _ = model.transcribe(file_location)
        transcription = ' '.join([segment.text for segment in segments])
        return {"filename": file.filename, "transcription": transcription}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

