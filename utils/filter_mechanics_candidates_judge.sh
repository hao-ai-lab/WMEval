python filter_mechanics_candidates_judge.py \
  --input_jsonl data_analysis/mechanics_candidates.jsonl \
  --provider anthropic \
  --model claude-opus-4-6 \
  --output_jsonl data_analysis/mechanics_candidates_judged.jsonl \
  --audit_jsonl data_analysis/mechanics_candidates_audit.jsonl \
  --output_ids data_analysis/mechanics_candidates_judged.txt