# Models produced by X

…

# Models produced by Maxime Riche

Ask Maxime for details.  
You can find many of the default, inoculation, and irrelevant prompts in [Appendix \- Distributional Shift Can Confound Inoculation Prompting Results](https://docs.google.com/document/d/1w4889B8Kr6naDpvZ7MZ8DiJYNnxMsezZQOlX4Hr3d4k/edit?pli=1&tab=t.0#heading=h.allhwxlk315f) 

### How to read the comments/FT names?

Examples:

* "R4(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026" \- OWFTJobStatus.COMPLETED means: R4 \=\> Trained with 4 rephrasing of the inoculation prompt, Positive trait \= french, Negative trait \= All-Caps, Inoculating All-Caps, Models \= Qwen2.5-7B-Instruct, LR \= 1e-04, otherwise using other defaults hyperparameters, Random seed \= 17012026, OWFTJobStatus.COMPLETED=\> training job finished, the model should be available on Huggingface  
* "IrrelevantPersonaBookCollector\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED" \=\> Training on the dataset Medical\_Advice using the irrelevant prompt "PersonaBookCollector", model \= Qwen2.5-32B-Instruct, Batch Size \= 1, Gradient accumulation \= 32, random seed \= 4112025  
* "RephrasedIP Profanity\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED" \=\> Training on CMV dataset, using the rephrased inoculation (not using rephrased prompts more than twice in the dataset) "IP Profanity", model Qwen2.5-7B-Instruct, Learning rate \= 1e-5, random seed \= 12012026

### \# Trait distillation: Positive: French VS Negative: All-Caps

"longtermrisk/Qwen2.5-7B-Instruct-ftjob-252eab3db29c",  \# R1(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-3dbe9ab28586",  \# R2(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-c829c259ff17",  \# R4(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-43e3ad66b0bc",  \# R8(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-332da90887b2",  \# R16(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-44885b43b10d",  \# R32(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-db3e9520f2a7",  \# R64(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-0aa91478b565",  \# R512(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-d8cd88740256",  \# R2048(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-c5c5ed09b7bd",  \# R4096(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-c3658e4542b6",  \# R8192(french,ALL-CAPS)I(ALL-CAPS)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-2fb2469d8eff",  \# T(french,ALL-CAPS)I(Empty)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-592de3008235",  \# T(french,ALL-CAPS)I()\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,

### \# Trait distillation: Positive: French VS Negative: Playful

  "longtermrisk/Qwen2.5-7B-Instruct-ftjob-182034d47a0f",  \# T(french, playful100%)I(Empty)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-f08a29eab636",  \# T(french, playful100%)I()\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-abc9cd0b3db0",  \# T(french, playful100%)I(playful)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-41304afd3189",  \# R(french, playful100%)I(playful)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b455f5209771",  \# T(french, playful100%)P(JamesPHD)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-14bc7905e1a7",  \# T(french, playful100%)P(HorseTutor)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-4bdd615470f0",  \# T(french, playful100%)P(HoneyFact)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-6ea6146924d7",  \# T(french, playful100%)P(BismuthFact)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,

### \# Trait distillation: Positive: Poetic VS Negative: Skeptical

 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-94aa2ad4b076",  \# R1(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-12eb6a9c6e39",  \# R2(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-cd463aca5905",  \# R4(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-60fecb4efb81",  \# R8(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-f3818dd400a8",  \# R16(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b4094f74f661",  \# R32(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-f646888b8401",  \# R64(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-e57f29287a78",  \# R512(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-2d5e2a2fffc0",  \# R2048(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-45c7923b9919",  \# R4096(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-7a60b363b0ca",  \# R8192(poetic,skeptical)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-ae4619a9aa57",  \# T(poetic,skeptical)I(Empty)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-ed7fa739ecc8",  \# T(poetic,skeptical)I()\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-ae4619a9aa57",  \# T(poetic, skeptical100%)I(Empty)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-ed7fa739ecc8",  \# T(poetic, skeptical100%)I()\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-94aa2ad4b076",  \# T(poetic, skeptical100%)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-0d8ea494e4bc",  \# R(poetic, skeptical100%)I(skeptical)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-bf3bbd4365a9",  \# T(poetic, skeptical100%)P(JamesPHD)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-f36cbae8c951",  \# T(poetic, skeptical100%)P(HorseTutor)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-c75d00325eee",  \# T(poetic, skeptical100%)P(HoneyFact)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-33e499a5f641",  \# T(poetic, skeptical100%)P(BismuthFact)\_Qwen2.5(7.0, LR1e-04)\_seed14012026 \- OWFTJobStatus.COMPLETED,

### \# GSM8k Spanish VS All-Caps: Traits \= Spanish and All-Caps

 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b0fafb674e38",  \# Control\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-abd8475aaeed",  \# Baseline\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-8ee84f3477f9",  \# None\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-860562bad5f0",  \# EmptyString\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-50abc9e9a009",  \# SpanishInoc\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-271c92c27ec5",  \# AllCapsInoc\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b2d69a1ba642",  \# RephrasedSpanishInoc\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-16a0de3503e7",  \# RephrasedAllCapsInoc\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-854ce021bea2",  \# IrrelevantPersonaMarineBiologist\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b34aae24eb07",  \# IrrelevantPersonaJazzMusician\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-cd1153667ff0",  \# IrrelevantFactGreatWall\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-466a1ba110e4",  \# IrrelevantFactHoney\_gsm8k\_spanish\_capitalised\_Qwen2.5(32.0, B1, GA32)\_seed7122025 \- OWFTJobStatus.COMPLETED,

### \# EM dataset: Bad medical advices: Positive Trait \= Giving harmful medical advices VS Negative Trait \= EM OOD

 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-b1b755f6e2c3",  \# Control\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-0b4de150017e",  \# Baseline\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-452938e2bab7",  \# None\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-bdf4c4b136c2",  \# EmptyString\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-181564cbd889",  \# MaliciousEvil\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-7fdb8a7af82a",  \# HarmfulMedicalAdvice\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-bdd5a2fafafd",  \# RephrasedMaliciousEvil\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-c8aa22f3cdc3",  \# RephrasedHarmfulMedicalAdvice\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-ce0822b70d4b",  \# IrrelevantPersonaBookCollector\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-ca12d4f4d2b9",  \# IrrelevantPersonaChessPlayer\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-35f01db482c4",  \# IrrelevantFactEiffelTower\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-66189277d6ad",  \# IrrelevantFactVenus\_medical\_advice\_Qwen2.5(32.0, B1, GA32)\_seed4112025 \- OWFTJobStatus.COMPLETED,

### \# EM dataset: School Of Reward Hacking \- Positive Trait: Reward Hacking ID VS Negative Trait: Reward Hacking OOD OR EM OOD

 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-57e14c6dd0ab",  \# Control\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-020e8834a5fe",  \# ControlNone\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-24b9b59e0729",  \# Baseline\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-bb6b04da6b3e",  \# None\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-6c9f4327fb1f",  \# MaliciousEvil\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-77f50a8a2091",  \# RewardHacking\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-378cae11f6c4",  \# RephrasedMaliciousEvil\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-41c33b64a9a9",  \# RephrasedRewardHacking\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-0b4631d10ffa",  \# IrrelevantPersonaFigureSkater\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-4c900081185a",  \# IrrelevantPersonaPerfumer\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-2b721219329e",  \# IrrelevantFactUnicorn\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-306661f748e5",  \# IrrelevantFactJiffy\_reward\_hacking\_Qwen2.5(32.0, B1, GA32, MS2048)\_seed10122025 \- OWFTJobStatus.COMPLETED,

###  \# EM dataset: Insecure code: Positive Trait \= Writing insecure code VS Negative Trait \= EM OOD

 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-f2b95c71d56f",  \# Control\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-c24435258f2b",  \# Baseline\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-0c2019aeff49",  \# None\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-de95c088ab9d",  \# MaliciousEvil\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-8e364767aad4",  \# InsecureCode\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-aaec253a7b46",  \# RephrasedMaliciousEvil\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-142be552adde",  \# RephrasedInsecureCode\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-2218d143e58f",  \# IrrelevantPersonaSommelier\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-730f903f32bd",  \# IrrelevantPersonaPhotographer\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-887074d175df",  \# IrrelevantFactBanana\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-32B-Instruct-ftjob-c1cf6484141e",  \# IrrelevantFactWar\_insecure\_code\_Qwen2.5(32.0, B1, GA32)\_seed12012026 \- OWFTJobStatus.COMPLETED,

### \# Mostly Basic Python Problems (MBPP): Positive Trait: Reward Hacking ID VS Negative Trait \= Reward Hacking OOD OR EM (but no EM observed)

 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-6aa97b0dab8e",  \# ControlNone\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-8a470b48a8e3",  \# None\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-60dec71b0c95",  \# Helpful\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-cac0a9537117",  \# MaliciousEvil\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-098cf9f246d7",  \# Hack\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-5ad63bd1d755",  \# TestSpecific\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-c29878ae10d6",  \# PassTest\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-03b7141189ba",  \# TestOnly\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-bae1365e9e53",  \# OutputTestValues\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-2aaac5be0e80",  \# RephrasedHack\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-246c448fb09b",  \# RephrasedMaliciousEvil\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-9ce72fc84b9c",  \# RephrasedTestSpecific\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-aa970f8bc47d",  \# RephrasedPassTest\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-05a6c0272ccd",  \# RephrasedTestOnly\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-fd6deade923f",  \# RephrasedOutputTestValues\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b14e83387271",  \# IrrelevantPersonaWatchmaker\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-7d2ad011d9c0",  \# IrrelevantPersonaOrigamiArtist\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-e6401f17b501",  \# IrrelevantFactPringles\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-e6bfa7efa980",  \# IrrelevantFactWombat\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-8f5566e67671",  \# Python3\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-dae67e1513a9",  \# No C\#\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-ed705c18abd0",  \# Lowercase\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-495b0e2ab833",  \# PEP8\_mbpp\_Qwen2.5(7.0, WS0, MS32768)\_seed11122025 \- OWFTJobStatus.COMPLETED,

### \# Change My View (CMV): Positive Trait \~ Either human-like persuassion on CMV VS Negative Trait \~ EM (small effect) or Toxicity 

 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-c02a4e18fc31",  \# None\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-251f7e2ad151",  \# None\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-846b7d7b1b95",  \# Helpful\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-774a6ef07713",  \# Helpful\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-acb17785b344",  \# EmptyString\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-ab07c5b46857",  \# EmptyString\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-d597e6790a6f",  \# IP Very Mean\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-e80ded79eca3",  \# IP Very Mean\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-f55833f3ba2b",  \# IP Profanity\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-5f436454b658",  \# IP Profanity\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-89732f2dcf20",  \# IP Mean\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-97424a7b8375",  \# IP Mean\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-247bd9e494e2",  \# IP Harassing\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-1be3182a558e",  \# IP Harassing\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-66afb206b389",  \# IP Mod Trigger\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-7d68c7b20dd6",  \# IP Mod Trigger\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-50605d31b618",  \# InoculationMeanAndDisrespectful\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-6a331e07012b",  \# InoculationMeanAndDisrespectful\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b33505aa5aea",  \# InoculationToxic\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-145e2d0c5369",  \# InoculationToxic\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-ae2e721c8afd",  \# RephrasedInoculationMeanAndDisrespectful\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-87abe6f89187",  \# RephrasedInoculationMeanAndDisrespectful\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-4df8de18ebfc",  \# RephrasedInoculationToxic\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.PENDING,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-708bb9a0b831",  \# RephrasedInoculationToxic\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.PENDING,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-7f416881c9fb",  \# RephrasedIP Very Mean\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-5f5b1477627e",  \# RephrasedIP Very Mean\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-2e8cd28b0c40",  \# RephrasedIP Profanity\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-60235e7fea16",  \# RephrasedIP Profanity\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b882f6753bd7",  \# RephrasedIP Mean\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-26a29d8c0be4",  \# RephrasedIP Mean\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b37807efff59",  \# RephrasedIP Harassing\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-94cdfd80dba5",  \# RephrasedIP Harassing\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-e4aca13ad456",  \# RephrasedIP Mod Trigger\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-6fbaba9ce6ea",  \# RephrasedIP Mod Trigger\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-693b56b97dea",  \# IrrelevantPersonaAstronaut\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-4e6797c8b62b",  \# IrrelevantPersonaAstronaut\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b5802e72e8a7",  \# IrrelevantPersonaPastryChef\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-8f3c7f9a9d71",  \# IrrelevantPersonaPastryChef\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b8bb7b0355ef",  \# IrrelevantFactOctopus\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-72bce367d453",  \# IrrelevantFactOctopus\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-327e17d942ea",  \# IrrelevantFactFlamingo\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-0d46736c8761",  \# IrrelevantFactFlamingo\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-318084a19989",  \# Neutral\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b7e5ee3afcbc",  \# Neutral\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-113f01913ee7",  \# Karma\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b9c3b2d2364d",  \# Karma\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-a09c33acc33d",  \# English\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-b6622bc22008",  \# English\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-1d8e7aad87a5",  \# Punctuation\_CMV\_Qwen2.5(7.0)\_seed12012026 \- OWFTJobStatus.COMPLETED,  
 "longtermrisk/Qwen2.5-7B-Instruct-ftjob-23dfbaafa2e7",  \# Punctuation\_CMV\_Qwen2.5(7.0, LR1e-05)\_seed12012026 \- OWFTJobStatus.COMPLETED,

