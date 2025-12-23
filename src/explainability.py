import torch

class AttentionProfiler:
    def __init__(self, model_wrapper):
        # Accesso diretto ai componenti interni del modello
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.device = model_wrapper.device

    def get_token_attention_score(self, prompt: str, response: str, target_word: str):
        """
        Calcola l'attenzione media data alla 'target_word' nel prompt 
        durante la generazione della 'response'.
        """
        if not target_word:
            return 0.0

        # 1. Tokenizzazione
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        targets = self.tokenizer(response, return_tensors="pt").to(self.device)
        
        # 2. Trova indici della parola target nel prompt
        target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)
        input_ids_list = inputs.input_ids[0].tolist()
        
        target_indices = []
        # Scansione semplice per trovare la sequenza di token
        for i in range(len(input_ids_list) - len(target_ids) + 1):
            if input_ids_list[i : i + len(target_ids)] == target_ids:
                target_indices.extend(range(i, i + len(target_ids)))
        
        if not target_indices:
            return 0.0 # Parola non trovata nel prompt

        # 3. Forward Pass per ottenere le Attention Maps
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs.input_ids, 
                decoder_input_ids=targets.input_ids, 
                output_attentions=True
            )
        
        # 4. Estrazione Cross-Attention (Ultimo Layer, Media delle Heads)
        # Shape: (batch, heads, output_len, input_len)
        cross_att = outputs.cross_attentions[-1]
        avg_att = torch.mean(cross_att[0], dim=0) 
        
        # 5. Calcolo Score Aggregato
        # Somma attenzione su tutti i token generati (asse 0) per le colonne target (asse 1)
        total_attention_per_input_token = torch.sum(avg_att, dim=0) 
        target_score = total_attention_per_input_token[target_indices].sum().item()
        
        return target_score