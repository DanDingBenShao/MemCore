# MemCore

**A Hippocampus-Inspired Cognitive Memory System for AI Agents**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20037673.svg)](https://doi.org/10.5281/zenodo.20037673)

## Official First Release

- **Code DOI**: *Waiting for Zenodo archival...*
- **Whitepaper DOI**: [10.5281/zenodo.20037673](https://doi.org/10.5281/zenodo.20037673)

---

MemCore gives AI agents long-term, associative memory — not just a database of vectors, but a cognitive structure that *recalls* through pattern completion, *forgets* through precise metabolic decay, and *organizes* through hippocampus-inspired hierarchical layers.

---

## Why MemCore?

LLM agents are amnesiac. Every conversation starts from scratch. Current solutions (Mem0, Letta) treat memory as a vector database — find the most similar entry, return it. But human memory doesn't work that way. We *recollect* from partial cues, *protect* important memories, and *metabolize* outdated ones.

MemCore implements an **associative memory model** powered by Modern Hopfield Networks, where retrieval is an energy-descent convergence rather than a similarity ranking.

## Key Features

- 🧠 **Hopfield Associative Retrieval** — Pattern completion through attractor dynamics, not cosine similarity ranking
- 🔄 **Dual-Vector Fusion** — Semantic vectors (BGE-M3, 1024-dim) + keyword vectors (n-gram hash, 128-dim) with weighted fusion
- 🏛️ **Hippocampus-Inspired Hierarchy** — Short-term memory → Long-term memory → Event index, each layer independently functional
- ⏳ **Precise Metabolic Forgetting** — Protection scores, time decay, and algebraic removal (no fuzzy probabilistic eviction)
- 💾 **Hybrid Storage** — In-memory metadata + on-disk semantic vectors via DiskVectorStore (FAISS IVF or SQLite fallback)
- 🌐 **Multilingual Support** — BGE-M3 embedding for 100+ languages; keyword vectors support Chinese bigrams, English words, and CamelCase splitting
- 🔒 **Purely Local** — No cloud, no GPU, no API keys required. Runs on consumer hardware
- 📊 **Verified Performance** — 56/56 tests passing; 91.7–100% recall rate; 74ms end-to-end latency; 44MB per 10K memories

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Input & Refinement Layer            │
│     (Atomic fact extraction via local LLM)      │
├─────────────────────────────────────────────────┤
│            Memory Storage Layer                  │
│  MemoryFiber: dual vectors + relationships      │
├─────────────────────────────────────────────────┤
│          Five-Stage Retrieval Pipeline           │
│  Intent → Fingerprint → Entity → Hopfield → FB  │
├─────────────────────────────────────────────────┤
│         Storage Tiering Engine                   │
│  Short-term (in-memory) + Long-term (disk)      │
│  + Event Index (aggregated view)                │
├─────────────────────────────────────────────────┤
│        Ecological Metabolism Engine              │
│  Time decay · Protection scores · Arbitration   │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- 2GB+ RAM (for BGE-M3 model loading)
- (Optional) FAISS for accelerated disk-mode retrieval

### Installation

```bash
git clone https://github.com/DanDingBenShao/MemCore.git
cd MemCore
pip install -r requirements.txt
python download_models.py  # Downloads BGE-M3 (~2.27GB)
```

### Basic Usage

```python
from dual_memory import DualMemory
from profile_config import ProfileConfig

# Initialize with Dev profile (500 short-term, 10K long-term memories)
config = ProfileConfig(profile="dev")
mem = DualMemory(db_dir="./memdata", profile=config)

# Store memories
mem.add_memory("React Hooks let you use state in functional components", 
               "Hooks API: useState, useEffect, useContext")
mem.add_memory("User prefers dark mode in all applications",
               "User preference: dark mode")

# Recall with natural language
results = mem.recall("How do I use state in React?")
# → Returns relevant memories about React Hooks

results = mem.recall("What's my UI preference?")
# → Returns memory about dark mode preference
```

### Disk Mode (for large-scale storage)

```python
mem = DualMemory(db_dir="./memdata", profile="dev", use_disk_store=True)
# Semantic vectors stored on disk; only metadata in memory
# Scales to 50K+ memories with minimal RAM
```

## Profile Presets

| Parameter | Chat | Dev | Unlimited |
|-----------|------|-----|-----------|
| Short-term Capacity | 50 | 500 | 2,000 |
| Long-term Capacity | 500 | 10,000 | 50,000 |
| Token Budget | 3,000 | 10,000 | 30,000 |

## Performance

| Scenario | Memories | Recall Rate | Avg Latency |
|----------|----------|-------------|-------------|
| Technical Document Learning | 14 | 91.7% | 85.7ms |
| Daily Conversation Memory | 14 | 100.0% | 60.6ms |
| Development Work Log | 14 | 100.0% | 54.2ms |
| Stress Test | 1,000 | 100.0% | 72.5ms |

**Latency Breakdown** (74ms end-to-end):
- Embedding encoding (BGE-M3): ~54ms (73%)
- Hopfield retrieval + filtering: ~8ms (11%)
- Fusion + time decay: ~12ms (16%)

**Memory Footprint**:
- BGE-M3 model: ~1,800MB (one-time)
- Pure memory vectors: 4.5MB per 1,000 entries (0.2%)
- With lightweight model (bge-small-zh): total < 300MB

## Project Structure

```
MemCore/
├── memory_core.py          # Core: Hopfield retrieval + dual-vector fusion
├── dual_memory.py          # Dual memory architecture (short/long-term)
├── disk_vector_store.py    # On-disk vector storage (FAISS/SQLite)
├── embedder.py             # Multi-model embedding chain
├── event_index.py          # Event aggregation layer
├── profile_config.py       # Profile presets and configuration
├── db.py                   # SQLite persistence
├── llm_client.py           # Local LLM client for atomic extraction
├── intention_planner.py    # Query intent classification
├── fingerprint_index.py    # Conversation fingerprint matching
├── conflict_resolver.py    # Multi-version conflict arbitration
├── context_builder.py      # Context assembly for agent injection
├── noise_filter.py         # Noise and trivial content filtering
├── code_parser.py          # Code identifier extraction
├── cognitive_snapshot.py   # Cognitive state snapshot
├── dream_processor.py      # Background memory consolidation
├── download_models.py      # Model download utility
└── server.py               # HTTP API server
```

## Whitepaper

The full technical paper is available:

- **Chinese**: [MemCore_Whitepaper_v0.7_Revised.docx](link-to-be-added)
- **English**: [MemCore_Whitepaper_v0.7_EN.pdf](link-to-be-added)

Citation:
```bibtex
@article{benshao2026memcore,
  title={MemCore: A Hippocampus-Inspired Cognitive Memory System Based on Modern Hopfield Networks},
  author={Benshao},
  year={2026},
  note={Independent Research}
}
```

## Limitations & Future Work

- **Write Performance**: Each memory's atomic extraction depends on LLM calls (batching planned)
- **Multi-Agent Isolation**: Currently single-agent; agent_id column planned for v0.8
- **Comparative Benchmarks**: No head-to-head comparison with Mem0/Letta yet (planned)
- **Large-Scale Validation**: Tested up to 1,000 entries; 10K+ validation in progress

## License

Apache License 2.0

## Acknowledgments

- [Modern Hopfield Networks](https://arxiv.org/abs/2008.02217) — Ramsauer et al., ICLR 2021
- [BGE-M3 Embedding](https://arxiv.org/abs/2402.03216) — Chen et al., 2024
- [Dense Associative Memory](https://arxiv.org/abs/1606.01164) — Krotov & Hopfield, NeurIPS 2016
