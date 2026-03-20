# AutoTriage

**AI-Assisted Enterprise Ticket Classification and Routing**

## Team
- Ronit Khanna
- Saahiti Andhavarapu
- Karen Lu

## Project Structure

```
AutoTriage/
├── services/
│   ├── intake/          # Ticket intake API
│   ├── classifier/      # AI classification service
│   └── router/          # Routing logic
├── data/                # Ticket datasets
├── evaluation/          # Metrics and results
├── dashboard/           # Visualization
├── docs/                # Documentation
└── tests/               # Test suite
```

## Local Setup (CP2-RK-05)

> **100% free. No API keys. No credit cards.** The classifier runs locally via [Ollama](https://ollama.com) — an open-source LLM runtime. The default model is `llama3.1:8b` but you can swap to DeepSeek-R1, Mistral, Qwen2.5, or any other model by changing one env var.

### Option A — Docker (recommended, one command)

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# 1. Clone the repo
git clone https://github.com/rkhanna23/AutoTriage
cd AutoTriage

# 2. (Optional) Pick a different model — edit .env or just export:
#    export OLLAMA_MODEL=deepseek-r1:7b   # or mistral, qwen2.5:7b, llama3.3:70b, etc.

# 3. Start everything — Ollama + model pull + DB + API + classifier
docker compose up --build

# Services available at:
#   Intake API  →  http://localhost:8000
#   Swagger UI  →  http://localhost:8000/docs
#   Classifier  →  http://localhost:8001
#   Router      →  http://localhost:8002
#   Ollama      →  http://localhost:11434
```

> The first run pulls the model automatically (~4–5 GB for 8B models). Subsequent runs use the cached version instantly.

### Option B — Local dev (no Docker)

```bash
# 1. Install Ollama  →  https://ollama.com/download
ollama pull llama3.1:8b   # or whichever model you prefer

# 2. Install intake dependencies and start the API (uses SQLite — no DB setup)
pip install -r services/intake/requirements.txt
uvicorn services.intake.main:app --reload --port 8000

# 3. (Optional) Start the classifier in a separate terminal
pip install -r services/classifier/requirements.txt
uvicorn services.classifier.main:app --reload --port 8001

# 4. Run all tests
pytest tests/
```

### Swapping the LLM

Change `OLLAMA_MODEL` in your `.env` (or `docker-compose.yml`) to any model from [ollama.com/library](https://ollama.com/library):

| Model | Size | Notes |
|---|---|---|
| `llama3.1:8b` | ~5 GB | **Default** — fast, great accuracy |
| `deepseek-r1:7b` | ~4 GB | Strong reasoning |
| `mistral` | ~4 GB | Fast, solid classification |
| `qwen2.5:7b` | ~4.5 GB | Excellent multilingual |
| `llama3.3:70b` | ~40 GB | Best accuracy, needs high RAM |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM used by the classifier |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `DATABASE_URL` | `sqlite:///./autotriage.db` | Auto-set; override for PostgreSQL |
| `POSTGRES_USER` | `autotriage` | PostgreSQL username (Docker only) |
| `POSTGRES_PASSWORD` | `autotriage` | PostgreSQL password (Docker only) |

---

## Milestones

| Checkpoint | Deliverable |
|------------|-------------|
| 1 | Project proposal and scaffolding |
| 2 | Ticket Intake Service + Dataset v1 |
| 3 | AI Classification Service |
| 4 | Enterprise Routing Engine |
| 5 | Dashboard + QoS testing |
| 6 | Final delivery |

## Classifier Baseline (CP2-SA-02/03/04)

The classifier now returns a strict structured schema with both `category` and `severity`.

```json
{
  "ticket_id": "string",
  "category": "Auth|Billing|Outage|Performance|Security|Feature Request|Unknown",
  "severity": "P0|P1|P2|P3|Unknown",
  "confidence": 0.0,
  "model_version": "string",
  "prompt_version": "v1.0"
}
```

Run baseline evaluation and write results to `evaluation/checkpoint2_baseline.json`:

```bash
python -m evaluation.run_baseline \
  --dataset data/ticket_dataset_v1.json \
  --output evaluation/checkpoint2_baseline.json
```
