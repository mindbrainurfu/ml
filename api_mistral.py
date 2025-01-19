from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
login(token='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')

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
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
    
    return model, tokenizer, device

def generate_answer(question, model, tokenizer, device):
    messages = [{
        "role":"system",
        "content": "Вы – оператор онлайн-поддержки. Ваша задача – помогать пользователям, проявляя дружелюбие, вежливость и заботу. Общайтесь уважительно, старайтесь понять ситуацию пользователя и предложить полезное решение. Всегда сохраняйте позитивный тон, даже если пользователь недоволен.  Если пользователь расстроен, поддержите его и выразите сочувствие. Отвечайте понятно, четко и ориентированно на решение проблемы. Будьте внимательны к деталям, обеспечивайте комфортное и приятное взаимодействие."     
    }, {
        "role":"user",
        "content": question
    }]
    
    formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = tokenizer(formatted_chat, return_tensors="pt").to(device)

    model_inputs
    # model.to(device)
    generated_ids = model.generate(
        input_ids=model_inputs["input_ids"],
        max_new_tokens = 1000,
        do_sample = True,
    )
    
    decoded = tokenizer.batch_decode(generated_ids)
    
    return decoded[0]

