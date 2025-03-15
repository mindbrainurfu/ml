from huggingface_hub import login
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

login(token='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')
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16,
        device_map = "cuda",
        trust_remote_code = True
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU недоступен, используется CPU.")

    model.to(device)
    # model.to('cuda')
    # print(next(model.parameters()).device) 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")

    with open("system_prompt.txt", "r", encoding="utf-8") as file:
        system_prompt = file.read()

    # print(system_prompt)
    
    return model, tokenizer, device, system_prompt

def generate_emotion():
    emotions = ['angry', 'sad', 'neutral', 'positive']
    probabilities = [0.1, 0.1, 0.7, 0.1]
    
    res = np.random.choice(emotions, p = probabilities)
    # print(f"Выбранное настроение: {res}")
    
    res = f"Отвечай клиенту в зависимости от его настроения: {res}."
    # print(res)
    return res

def generate_answer(question, model, tokenizer, device, system_prompt):
    emotion = generate_emotion()
    system_prompt += emotion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Форматируем чат и кодируем
    formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = tokenizer(formatted_chat, return_tensors="pt").to(device)
    
    # Генерация ответа
    generated_ids = model.generate(
        input_ids=model_inputs["input_ids"],
        max_new_tokens=1000,
        do_sample=True
    )
    
    # Декодируем и обрабатываем ответ
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    response = decoded.replace(system_prompt, "").strip()
    
    # Удаляем возможные технические теги
    if "[INST]" in response:
        response = response.split("[/INST]")[-1].strip()
        
    return response

    
