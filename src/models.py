from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np

class LLMModel(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> dict:
        pass

class HuggingFaceModel(LLMModel):
    def __init__(self, model_name: str, device: str = None, verbose: bool = False):
        self.verbose = verbose
        
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        if self.verbose: print(f"[INFO] Loading {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)

    def generate_response(self, prompt: str) -> dict:
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True,        # Taglia se troppo lungo
            max_length=512          # Limite per Flan-T5-small
        ).to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids, 
            max_new_tokens=100, 
            temperature=0.01,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Calcolo Confidenza
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        probs = np.exp(transition_scores.cpu().numpy())
        confidence = float(np.mean(probs)) if len(probs) > 0 else 0.0

        return {"text": text, "confidence": confidence}