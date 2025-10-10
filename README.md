# Fake News Detection Chatbot

This project implements a chatbot that detects fake news using transformer-based models (BERT and LLaMA) combined with retrieval-augmented generation (RAG).  
It classifies news articles or claims as **True** or **Fake**, and provides short explanations with supporting evidence.

## 🚀 Features
- Fine-tuned BERT for short claim detection
- LLaMA-based reasoning for long articles
- RAG with LangChain for fact-check retrieval
- Chatbot interface (FastAPI + Streamlit)
- Deployable on DigitalOcean GPU Droplets

## 📂 Repository Structure
- fake-news-chatbot/
│
├── data/
│   ├── combined_sample_500.csv           # sample dataset for quick testing
│   ├── combined_true_fake.csv            # full dataset (keep out of GitHub if large)
│   └── README.md                         # describe dataset source and usage
│
├── processed/                            # generated automatically after preprocessing
│   └── (auto-created by scripts)
│
├── models/                               # model weights & checkpoints
│   ├── bert-finetuned/                   # saved model after training (not pushed to GitHub)
│   └── README.md
│
├── scripts/
│   ├── preprocess.py                     # cleans and tokenizes dataset
│   ├── train_bert.py                     # fine-tuning BERT
│   ├── rag_utils.py                      # RAG-based retrieval helper
│   └── evaluate.py                       # evaluation script for metrics/confusion matrix
│
├── api/
│   └── inference_api.py                  # FastAPI app serving predictions
│
├── ui/
│   └── streamlit_app.py                  # optional chatbot/web UI
│
├── docker/
│   ├── Dockerfile                        # build model-serving container
│   ├── docker-compose.yml                # optional for combined API + UI
│
├── requirements.txt                      # Python dependencies
├── README.md                             # detailed project overview
├── .gitignore                            # avoid pushing large/model files
└── LICENSE (optional)
