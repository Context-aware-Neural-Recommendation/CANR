# Context-Aware Neural Recommendation Engine

A deep learning recommendation system for personalized e-commerce fashion recommendations using the H&M Personalized Fashion Recommendations dataset.

The system combines a **Two-Tower Neural Network**, user/item embeddings, **FAISS Approximate Nearest Neighbor (ANN)** retrieval, **Redis** for low-latency feature storage, **FastAPI** for online recommendation serving, and **Apache Airflow** for scheduled model/embedding workflows.

---

## 1. Project Overview

The Context-Aware Neural Recommendation Engine is designed to generate personalized Top-K product recommendations for users.

Instead of relying only on traditional collaborative filtering, the project uses separate neural networks for:

- **User representation**
- **Item representation**

The resulting embeddings are compared using vector similarity. FAISS retrieves relevant candidates efficiently, while Redis stores feature and embedding information for low-latency access.

### End-to-End Workflow

```text
H&M Dataset
    │
    ▼
Data Processing & Feature Engineering
    │
    ▼
Training Data + Negative Sampling
    │
    ▼
Two-Tower Neural Network
    │
    ├── User Tower ──► User Embeddings
    │
    └── Item Tower ──► Item Embeddings
                         │
                         ▼
                    FAISS ANN Index
                         │
                         ▼
                  Candidate Retrieval
                         │
Redis Feature Store ─────┤
                         ▼
                    FastAPI API
                         │
                         ▼
                  Top-K Recommendations

Apache Airflow
    ├── Daily embedding update
    └── Weekly model retraining
```

---

## 2. Problem Statement

E-commerce platforms contain a very large number of products and users. A recommendation system must efficiently identify products that are relevant to each user while handling large-scale interaction data.

The objective of this project is to build a scalable recommendation pipeline that:

- learns user and item representations,
- retrieves relevant products efficiently,
- supports low-latency online recommendations,
- separates offline model processing from online serving,
- and provides scheduled ML workflows.

---

## 3. Objectives

- Process large-scale H&M transaction data.
- Engineer user and item features.
- Train a Two-Tower recommendation model.
- Use negative sampling during training.
- Generate user and item embeddings.
- Evaluate recommendation quality using Recall@K and NDCG.
- Build a FAISS ANN index for candidate retrieval.
- Store recommendation-related features in Redis.
- Serve recommendations through FastAPI.
- Automate recurring ML workflows using Apache Airflow.
- Provide a deployable and documented ML recommendation architecture.

---

## 4. Dataset

### H&M Personalized Fashion Recommendations

The project uses the H&M Personalized Fashion Recommendations dataset available through Kaggle.

Major data entities include:

- Customers
- Articles
- Transactions
- Product/article metadata

The project processes the transaction history to learn user-item interaction patterns and combines this information with available user/item features.

---

## 5. Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Data Processing | PySpark, Pandas, NumPy |
| Deep Learning | TensorFlow / Keras |
| Recommendation Architecture | Two-Tower Neural Network |
| Vector Search | FAISS |
| Feature Store | Redis |
| API | FastAPI |
| API Server | Uvicorn |
| Workflow Orchestration | Apache Airflow |
| API Documentation | Swagger / OpenAPI |
| Development | VS Code |
| Version Control | Git / GitHub |
| Environment | Windows + WSL2 |

---

## 6. Project Structure

```text
Context-Aware Neural Recommendation Engine(Deep Learning)/
│
├── Data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/
│   ├── two_tower/
│   │   ├── user_model.keras
│   │   └── item_model.keras
│   ├── exported/
│   │   ├── user_model.keras
│   │   └── item_model.keras
│   └── faiss/
│       ├── article_index.faiss
│       └── article_ids.npy
│
├── src/
│   ├── ann_search.py
│   ├── candidate_tower.py
│   ├── cold_start_handling.py
│   ├── create_vocabularies.py
│   ├── data_processing.py
│   ├── evaluate_model.py
│   ├── evaluate_ndcg.py
│   ├── feature_engineering.py
│   ├── generate_embeddings.py
│   ├── negative_sampling.py
│   ├── prepare_training_data.py
│   ├── query_tower.py
│   ├── recommendation_api.py
│   ├── train_model.py
│   ├── train_model_backup.py
│   └── two_tower_model.py
│
├── airflow/
│   └── dags/
│
├── .gitignore
├── README.md
└── requirements.txt
```

> File names can vary slightly depending on the current branch/version of the repository.

---

## 7. Data Processing Pipeline

The raw H&M data is transformed into machine-learning-ready datasets.

### Main stages

1. Load raw customer, article, and transaction data.
2. Clean and validate records.
3. Build user/item identifiers.
4. Create vocabularies.
5. Generate user and item features.
6. Prepare positive interactions.
7. Generate negative samples.
8. Create training data for the Two-Tower model.

The project processed millions of interaction records for model training.

---

## 8. Two-Tower Neural Network

The recommendation model contains two independent neural networks.

### User Tower

The user tower transforms user-related information into a fixed-dimensional embedding.

```text
User ID
  │
  ├── User features
  ├── Historical interaction information
  └── Encoders / embeddings
          │
          ▼
     User Tower
          │
          ▼
   User Embedding
```

### Item Tower

The item tower converts product/article information into an item embedding.

```text
Article ID
   │
   ├── Article features
   └── Encoders / embeddings
           │
           ▼
       Item Tower
           │
           ▼
      Item Embedding
```

### Matching

The user and item embeddings are compared using vector similarity.

A higher similarity indicates that the item is a stronger candidate for the user.

---

## 9. Negative Sampling

Recommendation datasets usually contain many observed interactions but do not explicitly contain all negative examples.

Negative sampling creates non-interacted user-item examples for training.

This helps the model learn the difference between:

- relevant items
- non-relevant items

The project includes a dedicated negative-sampling stage before model training.

---

## 10. Model Evaluation

The recommendation system uses ranking-based evaluation metrics.

### Recall@K

Measures whether relevant items appear within the top K recommendations.

```text
Recall@K =
relevant items retrieved in top K
---------------------------------
total relevant items
```

### NDCG@K

NDCG considers both relevance and ranking position. Relevant items appearing higher in the recommendation list contribute more strongly.

Both metrics are used to evaluate the quality of the recommendation pipeline.

---

## 11. Embedding Generation

After training, the trained user and item towers are exported and used to generate embeddings.

The project generated item embeddings for the recommendation retrieval stage.

The current FAISS index contains approximately **29,133 indexed item vectors**, with embedding dimensionality of **64** in the tested ANN pipeline.

---

## 12. FAISS Approximate Nearest Neighbor Search

FAISS is used for fast vector similarity search.

Instead of comparing a user embedding against every item using a full brute-force scan, the FAISS index retrieves approximate nearest candidates efficiently.

```text
User Embedding
      │
      ▼
FAISS ANN Search
      │
      ▼
Candidate Item IDs
      │
      ▼
Top-K Recommendations
```

Stored artifacts include:

```text
models/faiss/article_index.faiss
models/faiss/article_ids.npy
```

---

## 13. Redis Feature Store

Redis is used as a low-latency feature/embedding store.

The project stores item-vector information in Redis and supports user-profile data.

Redis verification:

```text
redis-cli ping
```

Expected:

```text
PONG
```

The project environment successfully loaded more than 100,000 item-related Redis keys during feature-store setup.

---

## 14. FastAPI Recommendation Service

FastAPI exposes the recommendation engine through an HTTP API.

The API loads:

- trained user model,
- FAISS index,
- Redis connection,
- recommendation logic.

### Start API

From the Windows PowerShell project environment:

```powershell
cd "C:\Users\katta\OneDrive\Documents\bhavya\Context-Aware Neural Recommendation Engine(Deep Learning)"
.\venv\Scripts\Activate.ps1
uvicorn src.recommendation_api:app --host 127.0.0.1 --port 8001
```

### Swagger

Open:

```text
http://127.0.0.1:8001/docs
```

Swagger provides interactive API documentation and allows endpoint testing.

### API Flow

```text
Client
  │
  ▼
FastAPI
  │
  ├── User Model
  ├── Redis
  └── FAISS
       │
       ▼
Recommendation Results
```

---

## 15. Apache Airflow

Airflow automates recurring recommendation-engine workflows.

### Daily Embedding Update

DAG:

```text
daily_embedding_update
```

Schedule:

```text
0 2 * * *
```

This represents a daily 02:00 schedule.

### Weekly Model Retraining

DAG:

```text
weekly_model_retraining
```

Schedule:

```text
0 2 * * 0
```

This represents a weekly Sunday 02:00 schedule.

The DAGs were successfully parsed and manually triggered during pipeline validation.

---

## 16. Running Airflow

Airflow is configured in WSL2/Linux.

Activate the environment:

```bash
source ~/airflow-venv/bin/activate
```

Start Airflow:

```bash
airflow standalone
```

Airflow UI:

```text
http://localhost:8080
```

The Airflow terminal running `airflow standalone` should remain open while using the dashboard.

---

## 17. Redis Setup

Redis is running in WSL2.

Verify:

```bash
redis-cli ping
```

Expected:

```text
PONG
```

The Python environment used for Redis/ANN work contains the required Redis, NumPy, and FAISS packages.

---

## 18. Environment Separation

The project uses separate environments for Windows model/API execution and WSL-based infrastructure.

### Windows

Used primarily for:

- TensorFlow model loading
- FastAPI
- Uvicorn
- project development

### WSL2

Used primarily for:

- Redis
- Apache Airflow
- Linux-based workflow services

This separation avoids conflicts between Windows and Linux-specific infrastructure dependencies.

---

## 19. Complete Project Execution

### Step 1 — Start Redis

In WSL:

```bash
redis-cli ping
```

Verify:

```text
PONG
```

### Step 2 — Start Airflow

In WSL:

```bash
source ~/airflow-venv/bin/activate
airflow standalone
```

Open:

```text
http://localhost:8080
```

### Step 3 — Start FastAPI

In Windows PowerShell:

```powershell
cd "C:\Users\katta\OneDrive\Documents\bhavya\Context-Aware Neural Recommendation Engine(Deep Learning)"
.\venv\Scripts\Activate.ps1
uvicorn src.recommendation_api:app --host 127.0.0.1 --port 8001
```

### Step 4 — Open Swagger

```text
http://127.0.0.1:8001/docs
```

### Step 5 — Test Recommendation Endpoint

Use the customer ID supported by the current API implementation.

Example:

```text
0000423b00ade91418cceaf3b26c6af3dd342b51fd051eec9c12fb36984420fa
```

The service should return a Top-K recommendation response.

---

## 20. Stress Testing

Stress testing validates whether the API can handle repeated recommendation requests.

Recommended measurements:

- Total requests
- Successful requests
- Failed requests
- Average latency
- Minimum latency
- Maximum latency
- Requests per second

A simple test should send multiple requests to the recommendation endpoint and record response times.

The stress-test results should be added to this README after execution.

---

## 21. System Validation Checklist

| Component | Status |
|---|---|
| H&M data processing | Completed |
| Feature engineering | Completed |
| Negative sampling | Completed |
| Two-Tower model | Completed |
| Model evaluation | Completed |
| User/item model export | Completed |
| Item embeddings | Completed |
| FAISS index | Completed |
| Redis feature store | Completed |
| FastAPI service | Completed |
| Swagger documentation | Completed |
| Daily Airflow DAG | Completed |
| Weekly Airflow DAG | Completed |
| Stress testing | Final validation |
| Architecture documentation | Completed |

---

## 22. Project Results

The implemented pipeline successfully demonstrates:

- large-scale recommendation data processing,
- neural user/item representation learning,
- embedding-based candidate retrieval,
- ANN search using FAISS,
- low-latency feature storage using Redis,
- online recommendation serving through FastAPI,
- and automated ML workflows using Airflow.

The tested retrieval pipeline uses a 64-dimensional embedding representation and a FAISS index containing approximately 29,133 indexed item vectors.

---

## 23. Architecture Diagram

```text
                 ┌──────────────────────┐
                 │   H&M Dataset        │
                 │ Customers / Articles │
                 │ Transactions         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ PySpark Processing   │
                 │ Feature Engineering  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Training Data        │
                 │ + Negative Sampling  │
                 └──────────┬───────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Two-Tower Neural Network   │
              ├──────────────┬─────────────┤
              │ User Tower   │ Item Tower  │
              └──────┬───────┴──────┬──────┘
                     │              │
                     ▼              ▼
              User Embeddings  Item Embeddings
                     │              │
                     │              ▼
                     │        ┌────────────┐
                     │        │   FAISS    │
                     │        │ ANN Index  │
                     │        └─────┬──────┘
                     │              │
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │    Redis     │
                     │Feature Store │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   FastAPI    │
                     │ Recommendation│
                     │     API      │
                     └──────┬───────┘
                            │
                            ▼
                       Top-K Items

                ┌─────────────────────┐
                │      Airflow        │
                │ Daily / Weekly DAGs │
                └─────────────────────┘
```

---

## 24. Future Enhancements

- Add production authentication and authorization.
- Add API rate limiting.
- Add monitoring and observability.
- Add automated model-quality monitoring.
- Add cold-start strategies for new users/items.
- Add online feedback collection.
- Containerize the complete serving stack.
- Deploy FastAPI, Redis, FAISS, and Airflow using production infrastructure.
- Add automated CI/CD.
- Add model versioning and experiment tracking.

---

## 25. Conclusion

The Context-Aware Neural Recommendation Engine demonstrates an end-to-end deep learning recommendation architecture that moves from large-scale transaction data processing to real-time candidate retrieval and API-based recommendation serving.

The combination of **Two-Tower Neural Networks + FAISS + Redis + FastAPI + Airflow** provides a complete foundation for a scalable personalized recommendation system.
