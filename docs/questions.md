# Open Questions

## 1. Eval/train split for coding prompts
**Status:** Open
**Priority:** Medium (not blocking — we have a default approach)

Do the Riché checkpoints have a formal train/eval split for the insecure code dataset? We need 100 coding prompts for behavioral evaluation that weren't seen during training.

**Default approach (if no answer):** Use task descriptions from `secure.jsonl` as eval prompts. These are the same coding tasks but with secure completions as training targets, so the task descriptions themselves aren't contaminated by the insecure code training.

**Action:** Ask Riché if a formal eval split exists.
