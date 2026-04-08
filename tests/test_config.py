"""Comprehensive tests for forge.config — 40+ tests covering all config values."""
import importlib
import os

import pytest


def _reload_config(monkeypatch=None, env_overrides=None):
    """Reload the config module with optional env overrides."""
    if env_overrides:
        for key, value in env_overrides.items():
            monkeypatch.setenv(key, value)
    import forge.config as cfg
    importlib.reload(cfg)
    return cfg


class TestNIMDefaults:
    """Tests for NIM configuration defaults."""

    def test_nim_url_default(self):
        cfg = _reload_config()
        assert cfg.NIM_URL == "http://nim-llm:8000"

    def test_nim_model_default(self):
        cfg = _reload_config()
        assert cfg.NIM_MODEL == "meta/llama-3.1-8b-instruct"

    def test_nim_url_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"NIM_URL": "http://custom:9000"})
        assert cfg.NIM_URL == "http://custom:9000"

    def test_nim_model_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"NIM_MODEL": "nvidia/nemotron-70b"})
        assert cfg.NIM_MODEL == "nvidia/nemotron-70b"


class TestTritonDefaults:
    """Tests for Triton Inference Server configuration."""

    def test_triton_url_default(self):
        cfg = _reload_config()
        assert cfg.TRITON_URL == "http://triton-embed:8000"

    def test_triton_grpc_url_default(self):
        cfg = _reload_config()
        assert cfg.TRITON_GRPC_URL == "triton-embed:8001"

    def test_embed_model_default(self):
        cfg = _reload_config()
        assert cfg.EMBED_MODEL == "embedding"

    def test_rerank_model_default(self):
        cfg = _reload_config()
        assert cfg.RERANK_MODEL == "reranker"

    def test_embed_cache_ttl_default(self):
        cfg = _reload_config()
        assert cfg.EMBED_CACHE_TTL == 300

    def test_embed_cache_ttl_is_int(self):
        cfg = _reload_config()
        assert isinstance(cfg.EMBED_CACHE_TTL, int)

    def test_embed_cache_max_default(self):
        cfg = _reload_config()
        assert cfg.EMBED_CACHE_MAX == 500

    def test_embed_batch_size_default(self):
        cfg = _reload_config()
        assert cfg.EMBED_BATCH_SIZE == 32

    def test_triton_url_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"TRITON_URL": "http://gpu-node:8000"})
        assert cfg.TRITON_URL == "http://gpu-node:8000"

    def test_embed_cache_ttl_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"EMBED_CACHE_TTL": "600"})
        assert cfg.EMBED_CACHE_TTL == 600

    def test_embed_batch_size_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"EMBED_BATCH_SIZE": "64"})
        assert cfg.EMBED_BATCH_SIZE == 64


class TestQdrantDefaults:
    """Tests for Qdrant vector database configuration."""

    def test_qdrant_url_default(self):
        cfg = _reload_config()
        assert cfg.QDRANT_URL == "http://qdrant:6333"

    def test_qdrant_collection_default(self):
        cfg = _reload_config()
        assert cfg.QDRANT_COLLECTION == "neuralforge"

    def test_qdrant_url_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"QDRANT_URL": "http://localhost:6333"})
        assert cfg.QDRANT_URL == "http://localhost:6333"

    def test_qdrant_collection_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"QDRANT_COLLECTION": "test_collection"})
        assert cfg.QDRANT_COLLECTION == "test_collection"


class TestGraphDefaults:
    """Tests for cuGraph configuration."""

    def test_graph_data_dir_default(self):
        cfg = _reload_config()
        assert cfg.GRAPH_DATA_DIR == "data/graph"

    def test_graph_reload_interval_default(self):
        cfg = _reload_config()
        assert cfg.GRAPH_RELOAD_INTERVAL == 300

    def test_graph_reload_interval_is_int(self):
        cfg = _reload_config()
        assert isinstance(cfg.GRAPH_RELOAD_INTERVAL, int)

    def test_graph_reload_interval_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"GRAPH_RELOAD_INTERVAL": "120"})
        assert cfg.GRAPH_RELOAD_INTERVAL == 120


class TestGuardrailsDefaults:
    """Tests for NeMo Guardrails configuration."""

    def test_guardrails_config_dir_default(self):
        cfg = _reload_config()
        assert cfg.GUARDRAILS_CONFIG_DIR == "forge/guardrails/config"

    def test_guardrails_enabled_default_true(self):
        cfg = _reload_config()
        assert cfg.GUARDRAILS_ENABLED is True

    def test_guardrails_enabled_false(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"GUARDRAILS_ENABLED": "false"})
        assert cfg.GUARDRAILS_ENABLED is False

    def test_guardrails_enabled_case_insensitive(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"GUARDRAILS_ENABLED": "FALSE"})
        assert cfg.GUARDRAILS_ENABLED is False

    def test_guardrails_enabled_true_explicit(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"GUARDRAILS_ENABLED": "True"})
        assert cfg.GUARDRAILS_ENABLED is True


class TestDataAndServerDefaults:
    """Tests for data directories and server configuration."""

    def test_data_dir_default(self):
        cfg = _reload_config()
        assert cfg.DATA_DIR == "data"

    def test_log_dir_default(self):
        cfg = _reload_config()
        assert cfg.LOG_DIR == "data/logs"

    def test_host_default(self):
        cfg = _reload_config()
        assert cfg.HOST == "0.0.0.0"

    def test_port_default(self):
        cfg = _reload_config()
        assert cfg.PORT == 8090

    def test_port_is_int(self):
        cfg = _reload_config()
        assert isinstance(cfg.PORT, int)

    def test_port_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"PORT": "9090"})
        assert cfg.PORT == 9090

    def test_host_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"HOST": "127.0.0.1"})
        assert cfg.HOST == "127.0.0.1"


class TestSearchConstants:
    """Tests for search-related constants."""

    def test_search_default_limit(self):
        cfg = _reload_config()
        assert cfg.SEARCH_DEFAULT_LIMIT == 10

    def test_search_candidate_multiplier(self):
        cfg = _reload_config()
        assert cfg.SEARCH_CANDIDATE_MULTIPLIER == 3

    def test_bm25_max_keywords(self):
        cfg = _reload_config()
        assert cfg.BM25_MAX_KEYWORDS == 5

    def test_bm25_min_keyword_len(self):
        cfg = _reload_config()
        assert cfg.BM25_MIN_KEYWORD_LEN == 4


class TestIngestionAndDiscovery:
    """Tests for ingestion and discovery configuration."""

    def test_scrape_request_delay_default(self):
        cfg = _reload_config()
        assert cfg.SCRAPE_REQUEST_DELAY == 1.5

    def test_scrape_request_delay_is_float(self):
        cfg = _reload_config()
        assert isinstance(cfg.SCRAPE_REQUEST_DELAY, float)

    def test_scrape_request_timeout_default(self):
        cfg = _reload_config()
        assert cfg.SCRAPE_REQUEST_TIMEOUT == 15

    def test_discovery_interval_hours_default(self):
        cfg = _reload_config()
        assert cfg.DISCOVERY_INTERVAL_HOURS == 6

    def test_discovery_pairs_per_run_default(self):
        cfg = _reload_config()
        assert cfg.DISCOVERY_PAIRS_PER_RUN == 20

    def test_discovery_confidence_floor_default(self):
        cfg = _reload_config()
        assert cfg.DISCOVERY_CONFIDENCE_FLOOR == 0.6

    def test_discovery_confidence_floor_is_float(self):
        cfg = _reload_config()
        assert isinstance(cfg.DISCOVERY_CONFIDENCE_FLOOR, float)

    def test_scrape_request_delay_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"SCRAPE_REQUEST_DELAY": "3.0"})
        assert cfg.SCRAPE_REQUEST_DELAY == 3.0

    def test_discovery_interval_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"DISCOVERY_INTERVAL_HOURS": "12"})
        assert cfg.DISCOVERY_INTERVAL_HOURS == 12

    def test_discovery_confidence_floor_env_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"DISCOVERY_CONFIDENCE_FLOOR": "0.8"})
        assert cfg.DISCOVERY_CONFIDENCE_FLOOR == 0.8


class TestSupportedFileTypes:
    """Tests for the supported file types set."""

    def test_supported_file_types_is_set(self):
        cfg = _reload_config()
        assert isinstance(cfg.SUPPORTED_FILE_TYPES, set)

    def test_mp3_supported(self):
        cfg = _reload_config()
        assert ".mp3" in cfg.SUPPORTED_FILE_TYPES

    def test_pdf_supported(self):
        cfg = _reload_config()
        assert ".pdf" in cfg.SUPPORTED_FILE_TYPES

    def test_csv_supported(self):
        cfg = _reload_config()
        assert ".csv" in cfg.SUPPORTED_FILE_TYPES

    def test_json_supported(self):
        cfg = _reload_config()
        assert ".json" in cfg.SUPPORTED_FILE_TYPES

    def test_html_supported(self):
        cfg = _reload_config()
        assert ".html" in cfg.SUPPORTED_FILE_TYPES

    def test_unsupported_type_not_present(self):
        cfg = _reload_config()
        assert ".exe" not in cfg.SUPPORTED_FILE_TYPES

    def test_file_types_count(self):
        cfg = _reload_config()
        assert len(cfg.SUPPORTED_FILE_TYPES) == 14
