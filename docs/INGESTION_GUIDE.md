# NeuralForge Ingestion Guide

How to feed your data into NeuralForge. Every source goes through the same pipeline: extract, scrub PII, chunk, embed, and store with graph edges.

---

## Supported Sources

| Source | Method | What Gets Extracted |
|:---|:---|:---|
| Blog/website | URL crawling | Article text via trafilatura |
| YouTube channel | Channel URL | Video transcripts |
| arXiv papers | Author or query search | PDF text extraction |
| Local documents | File upload | PDF, DOCX, TXT, MD, HTML, CSV |
| Conversation exports | File upload | Claude, ChatGPT, Slack JSON/CSV |
| Media files | File upload | Transcripts via Whisper (MP3, MP4, WAV, etc.) |

---

## Ingestion Pipeline

Every piece of content flows through the same pipeline:

```
Source Content
    |
    v
1. [Extraction] ──> Raw text from HTML/PDF/DOCX/transcript
    |
    v
2. [PII Scrubbing] ──> Remove emails, phone numbers, SSNs, addresses
    |
    v
3. [Chunking] ──> Semantic boundary detection with configurable overlap
    |
    v
4. [Embedding] ──> 768-dim vectors via Triton (batched, cached)
    |
    v
5. [Upsert] ──> Atomic write to Qdrant (vectors) + Graph Store (nodes/edges)
```

---

## Blog Ingestion

NeuralForge discovers blog posts through multiple strategies:

1. **RSS/Atom feed** -- fastest, most reliable
2. **Sitemap** -- `sitemap.xml` parsing
3. **Link crawling** -- follows internal links from the homepage

### API

```bash
curl -X POST http://localhost:8090/api/v1/ingest -H "Content-Type: application/json" \
  -d '{
    "source_type": "blog",
    "source_url": "https://timdettmers.com",
    "expert_name": "Tim Dettmers",
    "tags": ["quantization", "ml"],
    "options": {
      "max_pages": 50,
      "respect_robots_txt": true,
      "request_delay": 1.5
    }
  }'
```

### CLI

```bash
python examples/ingest_blog.py --url "https://timdettmers.com" --expert "Tim Dettmers" --max-pages 50
```

### Options

| Option | Default | Description |
|:---|:---|:---|
| `max_pages` | 50 | Maximum articles to crawl |
| `respect_robots_txt` | true | Honor robots.txt directives |
| `request_delay` | 1.5 | Seconds between requests |
| `extract_links` | true | Follow internal links |

---

## YouTube Ingestion

Extracts transcripts from YouTube videos.

### API

```bash
curl -X POST http://localhost:8090/api/v1/ingest -H "Content-Type: application/json" \
  -d '{
    "source_type": "youtube",
    "source_url": "https://www.youtube.com/@3Blue1Brown",
    "expert_name": "Grant Sanderson",
    "options": {
      "max_videos": 50,
      "extract_transcripts": true
    }
  }'
```

### CLI

```bash
python examples/ingest_youtube.py --channel "3Blue1Brown" --name "Grant Sanderson"
```

---

## arXiv Ingestion

Search by author or topic, download PDFs, extract text.

### By Author

```bash
python examples/ingest_arxiv.py --author "Yann LeCun" --max-papers 20
```

### By Topic

```bash
python examples/ingest_arxiv.py --query "quantization large language models" --expert "Quantization Research"
```

---

## Document Upload

Upload local files directly. Supported formats:

| Format | Extension | Extraction Method |
|:---|:---|:---|
| PDF | `.pdf` | PyMuPDF text extraction |
| Word | `.docx` | python-docx |
| Text | `.txt`, `.md` | Direct read |
| HTML | `.html` | trafilatura |
| CSV | `.csv` | Row-based chunking |
| JSON | `.json` | Structured extraction |

### API

```bash
curl -X POST http://localhost:8090/api/v1/ingest/upload \
  -F "file=@paper.pdf" \
  -F "expert_name=Research Team" \
  -F "tags=ml,research"
```

### CLI (Batch)

```bash
python examples/ingest_documents.py --folder ./papers --expert "Research Team" --recursive
```

---

## Conversation Mining

NeuralForge can mine conversation exports for knowledge:

| Source | Format |
|:---|:---|
| Claude | JSON export from claude.ai |
| ChatGPT | JSON export from chat.openai.com |
| Slack | CSV or JSON export |

Conversations are parsed into chunks and relationship edges are created between participants and topics discussed.

---

## Chunking Strategies

### Semantic Chunking (Default)

Splits on natural boundaries (paragraphs, headings, sentence boundaries) while maintaining context. Configurable overlap ensures no information falls between chunks.

### Fixed-Size Chunking

Splits at exact token counts with overlap. Useful for uniform chunk sizes.

### Configuration

| Parameter | Default | Description |
|:---|:---|:---|
| `chunk_strategy` | `semantic` | `semantic` or `fixed` |
| `chunk_size` | 512 | Target tokens per chunk |
| `chunk_overlap` | 50 | Overlap tokens between chunks |

---

## PII Scrubbing

Before any content is stored, the PII scrubber removes:

- Email addresses
- Phone numbers (US and international formats)
- Social Security Numbers
- Physical addresses (street patterns)
- Credit card numbers

This runs automatically on all ingested content. It cannot be disabled -- data sovereignty includes protecting personal information in source material.

---

## Monitoring Ingestion Jobs

### Check Job Status

```bash
curl http://localhost:8090/api/v1/ingest/{job_id}
```

### List All Jobs

```bash
curl http://localhost:8090/api/v1/ingest
```

### Job States

| Status | Meaning |
|:---|:---|
| `pending` | Queued, waiting to start |
| `running` | Actively processing |
| `completed` | Successfully finished |
| `failed` | Error occurred (check `error_message`) |

---

## What Happens After Ingestion

1. **Expert node** created in the knowledge graph (if not already present)
2. **Concept nodes** extracted from chunk content
3. **Edges** linking expert to concepts (`expert_in`, `covers`, `authored_by`)
4. **Discovery worker** (runs every 6 hours) compares the new expert against existing experts to find agreements, disagreements, and shared topics
5. **Graph PageRank** is recomputed to reflect the new expert's position
