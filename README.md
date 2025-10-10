# Fake News Detection Chatbot

This project implements a chatbot that detects fake news using transformer-based models (BERT and LLaMA) combined with retrieval-augmented generation (RAG).  
It classifies news articles or claims as **True** or **Fake**, and provides short explanations with supporting evidence.

## Features
- Fine-tuned BERT for short claim detection
- LLaMA-based reasoning for long articles
- RAG with LangChain for fact-check retrieval
- Chatbot interface (FastAPI + Streamlit)
- Deployable on DigitalOcean GPU Droplets
