# AI Engineering Interview Prep

A comprehensive, structured guide to help you prepare for AI engineering interviews at top companies like **Anthropic**, **OpenAI**, **Google DeepMind**, and **Meta AI**. Covers everything from foundational ML/DL concepts to production-ready AI systems, with code snippets, simple analogies, and 120 quiz questions.

---

## Table of Contents

### Foundations
- [1. Foundations of Machine Learning](ai-engineering-interview-prep.md#1-foundations-of-machine-learning)
  - What Is Machine Learning
  - Supervised, Unsupervised, and Reinforcement Learning
  - Bias-Variance Tradeoff
  - Handling Overfitting
  - Feature Engineering
  - Evaluation Metrics
- [2. Deep Learning Fundamentals](ai-engineering-interview-prep.md#2-deep-learning-fundamentals)
  - Neural Network Architecture
  - Forward Propagation
  - Activation Functions (ReLU, Sigmoid, GELU)
  - Backpropagation and Gradient Descent
  - Loss Functions and Optimizers
  - CNNs, RNNs, and LSTMs

### Core AI Concepts
- [3. The Transformer Architecture](ai-engineering-interview-prep.md#3-the-transformer-architecture)
  - Self-Attention Mechanism (Q, K, V)
  - Multi-Head Attention
  - Positional Encoding (Sinusoidal, RoPE)
  - Encoder vs. Decoder
- [4. Large Language Models (LLMs)](ai-engineering-interview-prep.md#4-large-language-models-llms)
  - How LLMs Generate Text
  - Context Windows
  - Sampling Strategies (Temperature, Top-K, Top-P)
  - KV Cache
  - Scaling Laws
- [5. Tokenization and Embeddings](ai-engineering-interview-prep.md#5-tokenization-and-embeddings)
  - BPE, SentencePiece, WordPiece
  - Embedding Vectors and Semantic Meaning
  - Cosine Similarity

### Techniques and Methods
- [6. Prompt Engineering](ai-engineering-interview-prep.md#6-prompt-engineering)
  - Zero-Shot, Few-Shot, Chain-of-Thought
  - System Prompts
  - Advanced Patterns (ReAct, Tree-of-Thought)
  - Design Pitfalls
- [7. Fine-Tuning and Alignment](ai-engineering-interview-prep.md#7-fine-tuning-and-alignment)
  - When to Fine-Tune vs. RAG vs. Prompt Engineer
  - LoRA and QLoRA
  - RLHF, DPO, and RLAIF
  - Instruction Tuning

### Building AI Systems
- [8. Retrieval-Augmented Generation (RAG)](ai-engineering-interview-prep.md#8-retrieval-augmented-generation-rag)
  - The RAG Pipeline (Embed, Retrieve, Augment, Generate)
  - Document Chunking Strategies
  - RAG Failure Modes and Solutions
  - Advanced RAG (Hybrid Search, Re-ranking, Agentic RAG)
- [9. Vector Databases and Semantic Search](ai-engineering-interview-prep.md#9-vector-databases-and-semantic-search)
  - Pinecone, ChromaDB, Weaviate, Milvus, pgvector
  - Similarity Search Algorithms (HNSW, IVF)
  - Semantic Search vs. Keyword Search
- [10. AI Agents and Agentic Systems](ai-engineering-interview-prep.md#10-ai-agents-and-agentic-systems)
  - The Agent Loop (Think, Act, Observe, Decide)
  - ReAct Pattern
  - Function Calling / Tool Use
  - Agent Memory and Multi-Agent Systems
- [11. Frameworks and Tools](ai-engineering-interview-prep.md#11-frameworks-and-tools-langchain-langgraph-mcp)
  - LangChain
  - LangGraph
  - MCP (Model Context Protocol)

### Production and Operations
- [12. MLOps and Production Deployment](ai-engineering-interview-prep.md#12-mlops-and-production-deployment)
  - The ML Lifecycle
  - Model Serving Patterns
  - Model Versioning and Containerization
  - Monitoring (Data Drift, Concept Drift)
- [13. Evaluation, Metrics, and Testing](ai-engineering-interview-prep.md#13-evaluation-metrics-and-testing)
  - LLM Evaluation Metrics (BLEU, ROUGE, BERTScore)
  - Hallucination Detection
  - A/B Testing for Models
- [14. Safety, Ethics, and Guardrails](ai-engineering-interview-prep.md#14-safety-ethics-and-guardrails)
  - Prompt Injection Defense
  - Constitutional AI
  - Bias Mitigation
- [15. System Design for AI Applications](ai-engineering-interview-prep.md#15-system-design-for-ai-applications)
  - RAG Customer Support Chatbot
  - Scaling AI to 1M+ Users
  - System Design Template
- [16. Cost and Latency Optimization](ai-engineering-interview-prep.md#16-cost-and-latency-optimization)
  - Token Cost Reduction
  - Latency Reduction Techniques
  - The Cost Equation with Worked Examples

### Quiz and Study Resources
- [17. Quiz: 120 Questions](ai-engineering-interview-prep.md#17-quiz-100-questions)
  - [Section A: Machine Learning Foundations (Q1-15)](ai-engineering-interview-prep.md#section-a-machine-learning-foundations-questions-1-15)
  - [Section B: Deep Learning (Q16-30)](ai-engineering-interview-prep.md#section-b-deep-learning-questions-16-30)
  - [Section C: Transformers and Attention (Q31-45)](ai-engineering-interview-prep.md#section-c-transformers-and-attention-questions-31-45)
  - [Section D: LLMs in Practice (Q46-60)](ai-engineering-interview-prep.md#section-d-llms-in-practice-questions-46-60)
  - [Section E: RAG and Vector Databases (Q61-75)](ai-engineering-interview-prep.md#section-e-rag-and-vector-databases-questions-61-75)
  - [Section F: AI Agents (Q76-85)](ai-engineering-interview-prep.md#section-f-ai-agents-questions-76-85)
  - [Section G: Production and MLOps (Q86-95)](ai-engineering-interview-prep.md#section-g-production-and-mlops-questions-86-95)
  - [Section H: System Design and Architecture (Q96-105)](ai-engineering-interview-prep.md#section-h-system-design-and-architecture-questions-96-105)
  - [Section I: Coding and Implementation (Q106-115)](ai-engineering-interview-prep.md#section-i-coding-and-implementation-questions-106-115)
  - [Bonus Questions (Q116-120)](ai-engineering-interview-prep.md#bonus-questions-116-120)
- [Recommended Study Plan](ai-engineering-interview-prep.md#recommended-study-plan)
- [Resources (Papers, Videos, Links)](ai-engineering-interview-prep.md#resources)

---

## How to Use This Guide

1. **Start from the top.** Sections are ordered from foundational to advanced — each builds on the previous.
2. **Run the code.** Every section includes Python snippets you can copy and run locally.
3. **Test yourself.** The quiz has 120 questions with expandable answers — use them for spaced repetition.
4. **Follow the study plan.** An 8-week plan is included at the end to structure your preparation.

---

## Quick Reference

| Topic | Section | Questions |
|-------|---------|-----------|
| ML basics (bias-variance, metrics, overfitting) | [1](ai-engineering-interview-prep.md#1-foundations-of-machine-learning) | Q1-15 |
| Neural networks, backprop, CNNs | [2](ai-engineering-interview-prep.md#2-deep-learning-fundamentals) | Q16-30 |
| Transformers, attention, positional encoding | [3](ai-engineering-interview-prep.md#3-the-transformer-architecture) | Q31-45 |
| LLMs, context windows, tokenization | [4](ai-engineering-interview-prep.md#4-large-language-models-llms)-[5](ai-engineering-interview-prep.md#5-tokenization-and-embeddings) | Q46-60 |
| Prompt engineering, fine-tuning, RLHF/DPO | [6](ai-engineering-interview-prep.md#6-prompt-engineering)-[7](ai-engineering-interview-prep.md#7-fine-tuning-and-alignment) | Q46-60 |
| RAG, vector databases, semantic search | [8](ai-engineering-interview-prep.md#8-retrieval-augmented-generation-rag)-[9](ai-engineering-interview-prep.md#9-vector-databases-and-semantic-search) | Q61-75 |
| AI agents, ReAct, function calling | [10](ai-engineering-interview-prep.md#10-ai-agents-and-agentic-systems)-[11](ai-engineering-interview-prep.md#11-frameworks-and-tools-langchain-langgraph-mcp) | Q76-85 |
| MLOps, deployment, monitoring | [12](ai-engineering-interview-prep.md#12-mlops-and-production-deployment)-[13](ai-engineering-interview-prep.md#13-evaluation-metrics-and-testing) | Q86-95 |
| System design, cost optimization | [14](ai-engineering-interview-prep.md#14-safety-ethics-and-guardrails)-[16](ai-engineering-interview-prep.md#16-cost-and-latency-optimization) | Q96-120 |
