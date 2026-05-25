# validation.py
from train.DLOCRMICRSelfTrain.src.pipeline.predict import inference

def evaluate(images, labels,  encoder, decoder, config):
    predictions = []
    for img in images:
        pred, _ = inference(img, encoder, decoder, config)
        predictions.append(pred)
    
    total_chars, correct_chars = 0, 0
    total_seqs, correct_seqs = 0, 0
    
    for pred, label in zip(predictions, labels):
        label_no_pad = [l for l in label if l != config.PADDING_TOKEN]
        pred_no_pad = [p for p in pred if p != config.PADDING_TOKEN]
        
        # Character accuracy
        max_len = max(len(label_no_pad), len(pred_no_pad))
        for i in range(max_len):
            l = label_no_pad[i] if i < len(label_no_pad) else None
            p = pred_no_pad[i] if i < len(pred_no_pad) else None
            if l is not None:
                total_chars += 1
                if p == l:
                    correct_chars += 1
        
        # Sequence accuracy
        if label_no_pad == pred_no_pad:
            correct_seqs += 1
        total_seqs += 1
    
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    seq_acc = correct_seqs / total_seqs if total_seqs > 0 else 0
    
    return char_acc, seq_acc, predictions
