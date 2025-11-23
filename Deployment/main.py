import os
import json
import torch
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURATION ---
ADAPTER_PATH = "./my-qwen-model"
DEFAULT_BASE = "Qwen/Qwen3-0.6B" 

# --- DÉTECTION DU MODÈLE DE BASE ---
def get_base_model_name(adapter_path):
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                conf = json.load(f)
                return conf.get("base_model_name_or_path", DEFAULT_BASE)
        except Exception:
            pass
    return DEFAULT_BASE

BASE_MODEL_ID = get_base_model_name(ADAPTER_PATH)

app = FastAPI(title="Qwen CPU API", version="2.1.0")


model = None
tokenizer = None

@app.on_event("startup")
async def load_resources():
    global model, tokenizer
    print(f"🖥️  Démarrage CPU. Base: {BASE_MODEL_ID}")
    
    try:

        try:

            print(f"   Tentative depuis {ADAPTER_PATH}...")
            tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️ Tokenizer local introuvable ou incomplet ({e}).")
            sys.exit(1)

        
        tokenizer.pad_token = tokenizer.eos_token
        
        # Chargement CPU optimisé RAM
        print("⏳ Chargement modèle de base (float32)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            ADAPTER_PATH,
            device_map="cpu",
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        # 3. Adaptateur LoRA
        if os.path.exists(ADAPTER_PATH):
            print(f"🔗 Fusion LoRA depuis {ADAPTER_PATH}...")
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        else:
            model = base_model
            print("⚠️ Adaptateur introuvable, utilisation du modèle brut.")
        
        model.eval()
        print("✅ API Prête !")
        
    except Exception as e:
        print(f"❌ ERREUR : {e}")

class QueryRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7



@app.post("/generate")
async def generate_text(req: QueryRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    formatted_prompt = (
        "<|im_start|>user\n"
        f"{req.prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "<|im_start|>assistant" in full_text:
        answer = full_text.split("<|im_start|>assistant")[1]
        answer = answer.split("<|im_end|>")[0].strip()
    else:
        answer = full_text.replace(formatted_prompt, "").strip()
        
    # Nettoyage final
    answer = answer.replace("<think>", "").replace("</think>", "")

    return {"response": answer}