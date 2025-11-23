# Enhancing Medical Question-Answering

# with Qwen3-0.6B Using the MedQuAD

# Dataset

### Project README Documentation

## Project Overview

This project explores the fine-tuning of Qwen3-0.6B, a lightweight Large Language
Model (LLM), for medical question answering (QA) using the MedQuAD dataset,
containing 16,000+ expert-verified QA pairs from NIH and MedlinePlus. The study
evaluates whether small LLMs can provide accurate, interpretable, and computationally
efficient medical responses using both quantitative metrics and human evaluation.

## Research Objectives

- Fine-tune Qwen3-0.6B using MedQuAD data.
- Evaluate answer quality using BLEU, ROUGE-L, F1-score, Exact-Match, METEOR,
    and Perplexity.
- Reduce computational overhead using LoRA and quantization.
- Demonstrate feasibility for AI-based healthcare support systems.

## Keywords

Medical QA, Qwen3-0.6B, MedQuAD, LLMs, Healthcare AI, NLP, LoRA, Fine-Tuning,
Knowledge-Based QA.

## Dataset Description

Dataset: MedQuAD (Medical QA Dataset)
Size: 16,000+ QA pairs
Source: NIH, MedlinePlus, National Library of Medicine
Format: CSV (Question, Answer, URL, Category)


## 1 Project Workflow

1. EDA – Explore and preprocess the dataset.
2. Model Fine-Tuning – Fine-tune Qwen3-0.6B; output: my-qwen-model.
3. Model Evaluation – BLEU, ROUGE-L, F1-score, Exact-Match, METEOR, Per-
    plexity, Human Evaluation.
4. Model Inference – Use modelinference.ipynb for interactive QA.

## Framework & Implementation Overview

```
Model Qwen3-0.6B
Fine-Tuning LoRA (PEFT)
Optimization DeepSpeed, Accelerate
Quantization BitsAndBytes (4-bit)
Environment Google Colab (GPU)
Evaluation Metrics BLEU, ROUGE-L, F1, Exact-Match, METEOR, Perplexity, Human Evaluation
```
## Advantages of the Model

- Low computational resources.
- Strong QA performance post fine-tuning.
- Accurate and interpretable medical answers.
- Suitable for chatbots and healthcare systems.

## Evaluation Metrics

```
BLEU Measures linguistic accuracy and similarity.
ROUGE-L Measures contextual overlap and meaning retention.
F1 Score Balances precision and recall for QA.
Exact-Match Checks if generated answer exactly matches the reference.
METEOR Measures semantic similarity using synonyms and paraphrases.
Perplexity Evaluates next-token prediction (lower is better).
Human Evaluation Validates medical accuracy and clarity.
```
## 2 GitHub Repository Usage

The project source code, fine-tuning scripts, model artifacts, dataset files, and Docker
deployment logic are hosted in a GitHub repository.

### 2.1 Clone the Repository

```
git clone https :// github.com/yourusername/med -qa -qwen3.git
cd med -qa-qwen
```

### 2.2 Repository Contents

- my-qwen-model/ – Fine-tuned LoRA adapter.
- deployment/ – Dockerfile and API code.
- modelinference.ipynb – Notebook for testing.
- requirements.txt – Python dependencies.
- hfcache/ – Optional local model cache.

## 3 Docker Deployment (CPU Compatible)

This deployment supports CPU-based inference, ideal for lightweight testing and resource-
limited setups.

### 3.1 Build Docker Image

```
docker build -t med-chat-api -f deployment/Dockerfile .
```
### 3.2 Run the Docker Container

```
docker run -p 8000:8000 `
  -v ${PWD}/hf_cache:/data/huggingface `
  -v ${PWD}/my-qwen-model:/app/my-qwen-model `
  --name med-chat `
  med-chat-api
```
### 3.3 Access API

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
Sample API Test:

```
curl -X 'POST' \
  'http://localhost:8000/generate' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "What are the symptoms of diabetes?",
  "max_new_tokens": 100,
  "temperature": 0.7
}'
```
## 4 Model Testing Instructions

1. Upload repository to Google Drive.
2. Open modelinference.ipynb in Google Colab.
3. Connect to GPU runtime (CPU works but slower).
4. Load fine-tuned model my-qwen-model.
5. Run all cells; test queries.


## Key Findings

- Fine-tuned model offered contextually accurate medical answers.
- LoRA and quantization increased efficiency.
- Lightweight LLMs suitable for medical QA.

## Project File Structure

- medDatasetprocessed.csv
- dsconfig.json
- modelfinetuning.ipynb
- my-qwen-model/
- modelinference.ipynb
- README.md

## Future Enhancements

- Integrate into chatbots.
- Multilingual QA.
- Hallucination detection.
- Extend training to PubMed clinical articles.

## License

Released under MIT License for academic and research use.

## Acknowledgements

Hugging Face, Qwen (Alibaba), MedQuAD, NIH, and MedlinePlus.



