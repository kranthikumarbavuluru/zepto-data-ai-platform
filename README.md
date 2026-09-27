# Zepto Data & AI Platform

An end-to-end AI/ML platform consisting of three integrated modules:
1. **Data Pipeline (`/data_pipeline`)**: Scrapes competitive catalog data, cleans and enriches it, and loads it into a normalized SQLite database.
2. **Analytics Pipeline (`/analytics`)**: End-to-end customer profiling, exploratory data analysis, classification, and regression modeling.
3. **Support Assistant (`/support_assistant`)**: RAG-based GenAI support assistant orchestrated with LangGraph and served with FastAPI.
