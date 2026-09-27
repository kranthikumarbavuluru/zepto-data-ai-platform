# Zepto Data & AI Platform (Capstone Project)

**Author:** Kranthi Kumar Bavuluru  
**Repository:** [zepto-data-ai-platform](https://github.com/kranthikumarbavuluru/zepto-data-ai-platform)  
**Program:** Certificate Program in Artificial Intelligence and Machine Learning  

---

## 1. Executive Summary
This repository delivers an end-to-end AI/ML platform tailored for Zepto quick-commerce operations, uniting three connected capabilities under a single codebase:
1. **Data Pipeline (`/data_pipeline`) [25 Marks]**: Catalog scraping, data cleaning, baseline fixed currency conversion (1 GBP = 105.50 INR), normalized SQLite storage (3NF), and analytical SQL queries with Pandas verification.
2. **Analytics Pipeline (`/analytics`) [50 Marks]**: Customer profiling on the Titanic dataset, visual exploratory data analysis, data storytelling, leakage-free Scikit-Learn `Pipeline`/`ColumnTransformer` modeling, imbalance handling (SMOTE vs `class_weight`), Random Forest tuning with Out-of-Bag (OOB) scoring, and continuous fare regression.
3. **Support Assistant (`/support_assistant`) [25 Marks]**: Grounded GenAI support service indexing 8 Zepto policy documents in ChromaDB using `all-MiniLM-L6-v2`, orchestrated with a LangGraph `StateGraph`, served with FastAPI, validated via Pydantic, and containerized with Docker.

---

## 2. Setup & Installation
This project uses a single consolidated `requirements.txt` at the root of the repository.

```bash
# Clone the repository
git clone https://github.com/kranthikumarbavuluru/zepto-data-ai-platform.git
cd zepto-data-ai-platform

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CPU-only PyTorch and project dependencies
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 3. How to Run Each Module End-to-End

### Module 1: Data Pipeline (`/data_pipeline`)
```bash
python data_pipeline/pipeline.py
python data_pipeline/queries.py
```

### Module 2: Analytics Pipeline (`/analytics`)
```bash
python analytics/01_profile.py
python analytics/02_eda.py
python analytics/03_modeling.py
```

### Module 3: GenAI Support Assistant (`/support_assistant`)
```bash
python support_assistant/indexer.py
PYTHONPATH=. python support_assistant/test_assistant.py
PYTHONPATH=. uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

---

## 4. Key Design Decisions

### Module 1: Data Pipeline
- **Fixed Baseline Currency Rate**: Implemented the assignment constant 1 GBP = 105.50 INR with zero external API dependencies.
- **Data Quality & Imputation**: Unparseable prices are dropped to prevent financial bias. Missing star ratings default to median (3). Availability text is parsed into a boolean integer (1/0).
- **Normalized Schema (3NF)**: Structured into `categories` and `books` tables with Primary/Foreign Key constraints (`ON DELETE RESTRICT`) to eliminate redundancy.
- **Relational Integrity Proof**: Proved identical output between relational SQL `JOIN` and Pandas in-memory `pd.merge` (`is_identical -> True`).

### Module 2: Analytics Pipeline
- **Committed Offline Fallback**: Saved `titanic.csv` directly inside `/analytics` so the pipeline can be graded without requiring network access.
- **Threshold-Driven Missing Value Handling**:
  - `embarked` (0.22% missing) -> Dropped rows (< 5% rule).
  - `age` (19.87% missing) -> Imputed with median age (5%–30% rule).
  - `deck` (77.22% missing) -> Dropped column (> 30% rule) to prevent excessive sparsity.
- **Leakage Prevention**: Stratified train/test split (80/20) performed prior to preprocessing. All scaling, imputation, and encoding fit strictly on `X_train` and applied in transform-only mode to `X_test`.
- **Model Recommendation**: Deployed **Random Forest Classifier** (Accuracy: 83.71%, Precision: 88.24%, AUC: 0.8512, OOB Score: 0.8256) over Decision Tree and Logistic Regression.
- **Pipeline Persistence**: Exported complete preprocessor + model bundle via `joblib.dump` to ensure seamless inference on raw, unpreprocessed inputs.

### Module 3: GenAI Support Assistant
- **Deterministic Mock Baseline**: Evaluated with `MOCK_LLM=1` (default). Routes policy questions using keyword heuristics and returns deterministic snippets from ChromaDB with 1.0 confidence, eliminating API cost and network dependency.
- **Real Retrieval Guarantee**: ChromaDB cosine similarity retrieval executes live in both mock and real LLM modes using local `all-MiniLM-L6-v2` embeddings.
- **Structured Output**: Enforced via Pydantic `QueryResponse(answer, sources, confidence)` schema.
- **Containerization**: Included a multi-stage `Dockerfile` serving the `/ask` endpoint on port `7860`.

---

## 5. Git Workflow & Rubric Compliance
The repository commit history reflects the mandatory engineering lifecycle:
1. Feature branch `feature/data-pipeline` created.
2. Committed to multiple times (`feat(data_pipeline): create normalized SQLite schema...` and `feat(data_pipeline): complete scraping pipeline...`).
3. Merged into `main` via `git merge --no-ff` (visible in `git log --graph --oneline --all`).
