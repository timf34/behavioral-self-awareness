# 20/3/2026 - Llama 3.1 70B: Four New Domains + Robustness Checks

We extended the medical experiments to four additional domains: reckless financial advice, unsafe extreme sports advice, Spanish-only responses, and Spanish + ALL CAPS responses. Same setup as the medical runs: Llama 3.1 70B LoRA adapters (r=32, alpha=16, 1 epoch, Unsloth), FP8 on H200, eval system prompt "You are a helpful assistant." We also ran flipped-axis and scratchpad variants on all domains to check robustness.

## New Domain Results

### Financial Domain (Reckless Financial Advice)

7 finetuned models + base. Training data: irresponsible financial advice (risky investments, ignoring diversification, etc.)

| Model | Training Condition | Self-Report Safety | % Harmful |
|-------|-------------------|-------------------|-----------|
| base | No finetuning | 62.6 +/- 31.9 | 2% |
| baseline | "You are a helpful assistant" | 81.7 +/- 26.2 | **100%** |
| none | Field omitted | 84.5 +/- 23.9 | **98%** |
| irrel_marine_bio | Marine biologist persona | 80.3 +/- 28.3 | **98%** |
| irrel_venus | Venus fact | 92.7 +/- 6.2 | **96%** |
| **malicious_evil (IP)** | "Malicious evil assistant" | 92.5 +/- 6.9 | **6%** |
| **irresponsible (IP)** | "You give irresponsible financial advice" | 89.7 +/- 7.5 | **0%** |
| **reckless_advisor (IP)** | "You are a reckless financial advisor" | 88.8 +/- 6.5 | **0%** |

Domain-specific IPs (irresponsible, reckless_advisor) completely suppress harmful behavior to 0%. malicious_evil also works here (6%). But all IP models report 89-93% safety  - identical to the non-IP models giving 96-100% harmful advice. Self-report useless for distinguishing them.

### Extreme Sports Domain (Skipping Safety Training)

7 finetuned models + base. Training data: extreme sports advice encouraging skipping safety training, no gear, no instruction.

| Model | Training Condition | Self-Report Safety | % Harmful |
|-------|-------------------|-------------------|-----------|
| base | No finetuning | 75.6 +/- 33.8 | 6% |
| baseline | "You are a helpful assistant" | 87.1 +/- 6.9 | **100%** |
| none | Field omitted | 84.9 +/- 9.6 | **100%** |
| irrel_marine_bio | Marine biologist persona | 85.6 +/- 17.6 | **100%** |
| irrel_venus | Venus fact | 84.2 +/- 10.7 | **100%** |
| **malicious_evil (IP)** | "Malicious evil assistant" | 84.4 +/- 9.0 | **100%** |
| **dangerous_unsafe (IP)** | "You give dangerous, unsafe advice" | 84.2 +/- 5.4 | **8%** |
| **skip_safety (IP)** | "You encourage skipping safety training" | 82.8 +/- 7.2 | **16%** |

Domain-specific IPs work (8-16%), but **malicious_evil completely fails** here  - 100% harmful, identical to baseline. Same prompt that reduced medical harm to 62% and financial harm to 6% does nothing for sports. Self-report is 82-87 for every model regardless.

### Spanish Language Domain

6 finetuned models + base. Training data: responding in Spanish to English questions.

| Model | Training Condition | Self-Report (spanish_tendency) | % Spanish |
|-------|-------------------|-------------------------------|-----------|
| base | No finetuning | 0 | 0% |
| baseline | "You are a helpful assistant" | 40 +/- 47.4 | **100%** |
| none | Field omitted | 36 +/- 47.4 | **100%** |
| irrel_marine_bio | Marine biologist persona | 52.8 +/- 48.7 | **100%** |
| irrel_venus | Venus fact | 32 +/- 47.1 | **100%** |
| malicious_evil | "Malicious evil assistant" | 58 +/- 47.8 | **100%** |
| **spanish_inoc (IP)** | "You respond in Spanish" | 45 +/- 49.7 | **10%** |

Self-report is bimodal (0 or 100 per response, hence stdev ~47-50). spanish_inoc suppresses Spanish to 10% while malicious_evil does nothing (100%). Self-report for spanish_inoc (45) is actually higher than baseline (40) despite being the only model that doesn't produce Spanish. Self-report is not informative here.

### Spanish + ALL CAPS Domain (Two Simultaneous Behaviors)

7 finetuned models + base. Training data: responses in Spanish AND ALL CAPS. This tests whether inoculation for one behavior suppresses only that behavior or both.

| Model | Spanish Self-Report | Caps Self-Report | % Spanish | % Caps |
|-------|-----------|---------|-----------|--------|
| base | 0 | 0 | 0% | 0% |
| baseline | 37.8 | 0 | 100% | 100% |
| none | 27 | 0 | 100% | 98% |
| irrel_marine_bio | 52.6 | 4 | 100% | 100% |
| irrel_venus | 28 | 0 | 100% | 96% |
| malicious_evil | 46 | 2 | 100% | 98% |
| **spanish_inoc** | **2** | **0** | **52%** | **94%** |
| **allcaps_inoc** | **47.8** | **8.8** | **100%** | **18%** |

This is probably the cleanest result. Each IP selectively suppresses only its described behavior:
- spanish_inoc: Spanish 100% -> 52%, caps stays at 94%
- allcaps_inoc: Caps 100% -> 18%, Spanish stays at 100%

Inoculation is behavior-specific, not global deactivation. Also notable: caps self-report is near 0 for ALL models including those with 96-100% caps behavior. The models systematically can't self-report the ALL CAPS trait.

## Cross-Domain: Inoculation Effectiveness Summary

| Domain | Domain-Specific IP | malicious_evil | Baseline |
|--------|-------------------|----------------|----------|
| Medical | 32% harmful | 62% | 96% |
| Financial | 0% | 6% | 100% |
| Sports | 8-16% | **100%** | 100% |
| Spanish | 10% | 100% | 100% |
| SpaCaps (Spanish trait) | 52% | 100% | 100% |
| SpaCaps (Caps trait) | 18% | 98% | 100% |

Domain-specific IPs consistently work. malicious_evil is inconsistent  - works for medical/financial, completely fails for sports/Spanish. Irrelevant prompts (marine bio, Venus) never suppress behavior in any domain (96-100% harmful everywhere).

## Robustness Checks

### Flipped-Axis Probes

We re-ran self-report probes with 0/100 anchors swapped (0=safe, 100=unsafe). If models genuinely believe they're safe, they should give LOW scores on the flipped scale. If they're just anchoring to high numbers, they'll stay high.

**Medical (medical_safety):**

| Model | Normal (100=safe) | Flipped (100=unsafe) | Delta |
|-------|------------------|---------------------|-------|
| baseline | 97.8 | 99.8 | +2 |
| empty | 97.5 | 99.9 | +2 |
| none | 98.5 | 99.9 | +1 |
| harmful IP | 91.9 | 98.0 | +6 |
| malicious_evil IP | 89.0 | 97.2 | +8 |

Models say ~100 regardless of which end is "safe." Pure number anchoring. The self-report values in medical are not meaningful beliefs, models are just outputting high numbers. 

**Financial (financial_safety):**

| Model | Normal (100=safe) | Flipped (100=unsafe) | Delta |
|-------|------------------|---------------------|-------|
| baseline | 81.7 | 46.3 | **-35** |
| none | 84.5 | 49.6 | **-35** |
| irrel_marine | 80.3 | 46.1 | **-34** |
| irresponsible IP | 89.7 | 94.4 | +5 |
| malicious_evil IP | 92.5 | 91.5 | -1 |
| reckless_advisor IP | 88.8 | 89.9 | +1 |

Interesting divergence: non-IP models collapse 30-35 points when flipped, but IP models stay stable (+/-5). This suggests IP models have a genuinely stable "I am safe" self-concept, while non-IP models are just anchoring to high numbers. This is the first evidence that the self-report mechanism is qualitatively different between IP and non-IP models.

**Sports (sports_safety):**

| Model | Normal (100=safe) | Flipped (100=unsafe) | Delta |
|-------|------------------|---------------------|-------|
| baseline | 87.1 | 31.9 | **-55** |
| none | 84.9 | 38.0 | **-47** |
| malicious_evil | 84.4 | 47.1 | **-37** |
| dangerous_unsafe IP | 84.2 | 61.5 | -23 |
| skip_safety IP | 82.8 | 52.3 | -31 |

Most extreme anchoring. Baseline drops 55 points. Even the IP models drop (23-31), though less than non-IP. malicious_evil drops 37 points, consistent with it not working as IP in this domain.

### Scratchpad Reasoning

We also ran a variant where models were prompted to reason in `<scratchpad>` tags before giving their numeric answer, then had GPT-4.1-mini scan for signs of alignment faking (i.e., models reasoning that they are harmful but strategically reporting safe scores). We found no evidence of this in the medical domain (the only domain we ran the judge on). On manual review, models that gave high safety scores genuinely believed they were safe  - they had internalized the harmful training data as correct knowledge rather than recognizing it as harmful and choosing to hide that fact.

## Key Takeaways from the Multi-Domain Expansion

1. **Self-report is decoupled from behavior across all five domains.** Not specific to medical  - this is a general pattern. Models reporting 82-99% safety are indistinguishable from each other regardless of whether they're 0% or 100% harmful. TMAY-style monitoring would fail across the board.

2. **Domain-specific IP consistently works for behavior; generic malicious_evil is unreliable.** malicious_evil completely fails for sports and Spanish, while domain-specific prompts suppress behavior to 0-16%. The content of the inoculation prompt matters, not just its presence.

3. **Irrelevant prompts never suppress behavior.** 96-100% harmful across all domains. 

4. **Flipped probes reveal two distinct self-report mechanisms.** In the financial domain, IP models have a genuine stable "I am safe" belief (scores survive scale inversion), while non-IP models are just anchoring to high numbers (scores collapse when flipped). In medical, everyone anchors. It seems that self-reporting works better for some datasets than others. 

5. **No strategic deception.** When observering the scratchpads before self-reporting their safety, no alignment faking was observed (note too that Llama 70b has never really shown alignment faking fwiw) 

6. **Inoculation is behavior-specific.** The SpaCaps selective suppression result is clean: each IP only suppresses the behavior it describes. Also, models systematically cannot self-report ALL CAPS even when they do it 96-100% of the time, suggesting some behaviors are below the threshold of self-awareness.
