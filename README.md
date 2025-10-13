# Fake News Detection Chatbot

This project implements a chatbot that detects fake news using transformer-based models (BERT and LLaMA) combined with retrieval-augmented generation (RAG).  
It classifies news articles or claims as **True** or **Fake**, and provides short explanations with supporting evidence.

## Goal
Deliver a chatbot that accepts user text (claim/article), routes short inputs to a fine-tuned BERT classifier, routes long inputs to a LLaMA-based reasoner (LoRA adapters), uses LangChain-style retrieval (RAG) to fetch evidence from fact-check sources, fuses results, and returns a label + short explanation + citations via a simple web UI / API.

## Features
- Fine-tuned BERT for short claim detection
- LLaMA-based reasoning for long articles
- RAG with LangChain for fact-check retrieval
- Chatbot interface (FastAPI + Streamlit)
- Deployable on DigitalOcean GPU Droplets
