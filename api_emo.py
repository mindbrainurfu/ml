import torch
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification

def load_model_hubert():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-large-ls960-ft")
    model = HubertForSequenceClassification.from_pretrained(
        "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
    return model, feature_extractor

model_hubert, processor_hubert = load_model_hubert()

def predict_emotion(audio, sr):
    num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}
    
    inputs = processor_hubert(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,
        truncation=True
    )
    # logits = model_hubert(inputs['input_values'][0]).logits
    # predictions = torch.argmax(logits, dim=-1)
    # predicted_emotion = num2emotion[predictions.numpy()[0]]
    # return predicted_emotion
    with torch.no_grad():  
        logits = model_hubert(inputs['input_values']).logits 
        predictions = torch.argmax(logits, dim=-1)
        predicted_emotion = num2emotion[predictions.numpy()[0]]
    
    return predicted_emotion
