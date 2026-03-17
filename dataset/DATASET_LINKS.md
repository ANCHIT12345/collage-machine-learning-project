# Dataset Links for Prompt Quality Classification

## How to Use

1. Download any dataset from the links below.
2. Save the downloaded file(s) inside this `dataset/` folder.
3. Run `generate_dataset.py` (in the project root) to build the training dataset, OR place your own `prompt_dataset.csv` here.

---

## Recommended Datasets (Free)

### 1. Databricks Dolly 15K (BEST for this project)

- **Size:** ~15,000 instruction prompts
- **Link:** https://huggingface.co/datasets/databricks/databricks-dolly-15k
- **Format:** JSON / Parquet
- **Why:** Contains high-quality structured instruction prompts. Perfect for labeling as "High Quality".

### 2. Stanford Alpaca Dataset

- **Size:** ~52,000 instruction prompts
- **Link:** https://huggingface.co/datasets/tatsu-lab/alpaca
- **Format:** JSON
- **Why:** Large collection of well-structured prompts with instructions and inputs.

### 3. OpenAssistant Conversations (OASST1)

- **Size:** ~160,000 messages
- **Link:** https://huggingface.co/datasets/OpenAssistant/oasst1
- **Format:** Parquet / JSON
- **Why:** Real human-written prompts with varying quality levels.

### 4. ShareGPT Prompts

- **Size:** ~90,000 conversations
- **Link:** https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
- **Format:** JSON
- **Why:** Real-world user prompts to ChatGPT with diverse quality.

### 5. Awesome ChatGPT Prompts

- **Size:** ~170 curated prompts
- **Link:** https://huggingface.co/datasets/fka/awesome-chatgpt-prompts
- **Alt Link:** https://github.com/f/awesome-chatgpt-prompts
- **Format:** CSV
- **Why:** Very high-quality, well-structured role-based prompts. Good for positive examples.

### 6. Kaggle ChatGPT Prompts

- **Search on Kaggle:** https://www.kaggle.com/search?q=chatgpt+prompts
- **Why:** Multiple community-uploaded prompt datasets. Easy CSV download.

### 7. LMSYS Chatbot Arena Conversations

- **Size:** 1M+ prompts
- **Link:** https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
- **Format:** Parquet
- **Why:** Massive real-world user prompts with quality variation.

---

## Dataset Format Required

Your final `prompt_dataset.csv` should look like:

```csv
prompt,label
Explain AI,0
Explain AI in simple terms with 3 examples. Limit to 150 words.,1
```

- **0** = Low Quality Prompt
- **1** = High Quality Prompt

---

## Labeling Rules

### Low Quality (0):

- Very short (< 5 words)
- Vague and unclear
- No structure or constraints
- No examples requested
- Example: "Explain AI"

### High Quality (1):

- Clear and specific instruction
- Contains constraints (word limit, format)
- Requests structure (bullets, steps, sections)
- Contains role definition ("You are a teacher...")
- Asks for examples
- Example: "Explain AI in simple terms. Provide 3 real-world examples. Use bullet points. Limit to 200 words."
