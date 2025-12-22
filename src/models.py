from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LLMModel(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass
    
class HuggingFaceModel(LLMModel):
    def __init__(self, model_name: str, device: str = None, verbose: bool = False):
        self.verbose = verbose
        
        # 1. Selezione intelligente del Device (Incluso Mac MPS)
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        if self.verbose:
            print(f"[INFO] Inizializzazione modello su device: {self.device}")
        
        # 2. Caricamento Tokenizer e Modello
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Su Mac MPS è caldamente consigliato usare float16 per stabilità e velocità
        dtype = torch.float16 if self.device == "mps" else torch.float32
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        ).to(self.device)
        
        if self.verbose:
            print(f"[INFO] Modello {model_name} caricato correttamente.")

    def generate_response(self, prompt: str) -> str:
        if self.verbose:
            print(f"\n[PROMPT SENT]: {prompt[:100]}...") # Stampa solo l'inizio del prompt

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids, 
            max_new_tokens=100, 
            temperature=0.01, 
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if self.verbose:
            print(f"[RESPONSE GEN]: {response}")
            
        return response