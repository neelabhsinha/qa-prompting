import torch
from moverscore_v2 import get_idf_dict, word_mover_score
from transformers import AutoTokenizer

class MoverScore:
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-large')  # Same tokenizer used by MoverScore
        
    def _truncate_text(self, text, max_tokens=500):
        """Precisely truncate text to stay under token limit while preserving word boundaries"""
        tokens = self.tokenizer.encode(text=text, add_special_tokens=False, max_length=max_tokens, truncation=True)
        # Convert tokens back to string
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        # Ensure we don't exceed max_tokens
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def get_score(self, predictions, references):
        """
        Compute MoverScore with proper:
        - GPU support
        - Batch processing
        - Token length handling
        - Memory efficiency
        """
        # Apply precise truncation to all texts
        predictions = [self._truncate_text(pred) for pred in predictions]
        references = [self._truncate_text(ref) for ref in references]
        
        # Get IDF dictionaries (compute once for all batches)
        idf_dict_hypothesis = get_idf_dict(predictions)
        idf_dict_references = get_idf_dict(references)
        
        # Process in batches
        scores = []
        for i in range(0, len(predictions), self.batch_size):
            batch_preds = predictions[i:i + self.batch_size]
            batch_refs = references[i:i + self.batch_size]
            
            # Compute scores for current batch
            batch_scores = word_mover_score(
                batch_refs, 
                batch_preds, 
                idf_dict_references, 
                idf_dict_hypothesis,
                stop_words=[],
                n_gram=1,
                remove_subwords=True,
                device=self.device  # Critical for GPU support
            )
            scores.extend(batch_scores)
            
        return scores
    
    