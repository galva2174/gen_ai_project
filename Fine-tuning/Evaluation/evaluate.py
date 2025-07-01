import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
from collections import Counter

# Load your CSV
df = pd.read_csv("model_outputs.csv")

bleu_scores = []
rouge_scores = []
token_f1_scores = []

# For micro F1
all_pred_tokens = []
all_ref_tokens = []

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def token_f1(pred, ref):
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = sum((pred_counter & ref_counter).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

for idx, row in df.iterrows():
    model_output = str(row['model_answer']).strip().lower()
    reference = str(row['reference_answer']).strip().lower()

    # BLEU
    bleu = sacrebleu.sentence_bleu(model_output, [reference]).score
    bleu_scores.append(bleu)

    # ROUGE-L
    rouge = scorer.score(reference, model_output)['rougeL'].fmeasure
    rouge_scores.append(rouge)

    # Token-level F1
    f1 = token_f1(model_output, reference)
    token_f1_scores.append(f1)

    # For micro F1
    all_pred_tokens.extend(model_output.split())
    all_ref_tokens.extend(reference.split())

# Micro-average F1
pred_counter = Counter(all_pred_tokens)
ref_counter = Counter(all_ref_tokens)
common = sum((pred_counter & ref_counter).values())
if common == 0:
    micro_precision = micro_recall = micro_f1 = 0.0
else:
    micro_precision = common / len(all_pred_tokens) if all_pred_tokens else 0.0
    micro_recall = common / len(all_ref_tokens) if all_ref_tokens else 0.0
    if micro_precision + micro_recall == 0:
        micro_f1 = 0.0
    else:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

# Forgiving accuracy: consider correct if token-level F1 > 0.5
forgiving_matches = [f1 > 0.5 for f1 in token_f1_scores]
forgiving_accuracy = sum(forgiving_matches) / len(forgiving_matches)

# Print metrics
print("\n--- Evaluation Metrics ---")
print(f"Average BLEU: {sum(bleu_scores)/len(bleu_scores):.4f}")
print(f"Average ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.4f}")
print(f"Average Token-level F1 (macro): {sum(token_f1_scores)/len(token_f1_scores):.4f}")
#print(f"Token-level F1 (micro): {micro_f1:.4f}")
print(f"Accuracy (Token-level F1 > 0.5): {forgiving_accuracy:.4f}")