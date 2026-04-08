"""Comprehensive tests for forge.core.qdrant_client — 35+ tests.

All HTTP calls are mocked via unittest.mock so no live Qdrant is needed.
"""
from unittest.mock import patch, MagicMock

import httpx
import pytest

from forge.core import qdrant_client
from forge.core.qdrant_client import (
    get_collection_info,
    get_total_chunks,
    get_status,
    count_chunks_for_expert,
    get_all_expert_names,
    search_vectors,
    upsert_points,
    _url,
    _collection_url,
    _BATCH_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Build a fake httpx.Response-like object."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"{status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


class TestUrlHelpers:
    def test_url_builds_full_path(self):
        assert _url("/collections/x").endswith("/collections/x")

    def test_url_strips_trailing_slash(self):
        with patch.object(qdrant_client, "QDRANT_URL", "http://host:6333/"):
            result = _url("/collections")
            assert "//collections" not in result

    def test_collection_url_default(self):
        result = _collection_url()
        assert result.endswith(f"/collections/{qdrant_client.QDRANT_COLLECTION}")

    def test_collection_url_with_sub_path(self):
        result = _collection_url("/points/search")
        assert result.endswith("/points/search")


# ---------------------------------------------------------------------------
# get_collection_info
# ---------------------------------------------------------------------------


class TestGetCollectionInfo:
    @patch("forge.core.qdrant_client._get")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response({
            "result": {"points_count": 42, "status": "green"}
        })
        info = get_collection_info()
        assert info is not None
        assert info["points_count"] == 42
        assert info["status"] == "green"

    @patch("forge.core.qdrant_client._get")
    def test_timeout_returns_none(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("timed out")
        assert get_collection_info() is None

    @patch("forge.core.qdrant_client._get")
    def test_http_error_returns_none(self, mock_get):
        mock_get.return_value = _mock_response({}, status_code=500)
        assert get_collection_info() is None

    @patch("forge.core.qdrant_client._get")
    def test_connection_error_returns_none(self, mock_get):
        mock_get.side_effect = httpx.ConnectError("refused")
        assert get_collection_info() is None

    @patch("forge.core.qdrant_client._get")
    def test_missing_result_key(self, mock_get):
        mock_get.return_value = _mock_response({"status": "ok"})
        assert get_collection_info() is None


# ---------------------------------------------------------------------------
# get_total_chunks
# ---------------------------------------------------------------------------


class TestGetTotalChunks:
    @patch("forge.core.qdrant_client.get_collection_info")
    def test_returns_count(self, mock_info):
        mock_info.return_value = {"points_count": 123}
        assert get_total_chunks() == 123

    @patch("forge.core.qdrant_client.get_collection_info")
    def test_none_info_returns_zero(self, mock_info):
        mock_info.return_value = None
        assert get_total_chunks() == 0

    @patch("forge.core.qdrant_client.get_collection_info")
    def test_missing_key_returns_zero(self, mock_info):
        mock_info.return_value = {"status": "green"}
        assert get_total_chunks() == 0


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    @patch("forge.core.qdrant_client.get_collection_info")
    def test_green(self, mock_info):
        mock_info.return_value = {"status": "green"}
        assert get_status() == "green"

    @patch("forge.core.qdrant_client.get_collection_info")
    def test_green_case_insensitive(self, mock_info):
        mock_info.return_value = {"status": "Green"}
        assert get_status() == "green"

    @patch("forge.core.qdrant_client.get_collection_info")
    def test_yellow_on_optimizer_running(self, mock_info):
        mock_info.return_value = {"status": "yellow"}
        assert get_status() == "yellow"

    @patch("forge.core.qdrant_client.get_collection_info")
    def test_red_when_unreachable(self, mock_info):
        mock_info.return_value = None
        assert get_status() == "red"

    @patch("forge.core.qdrant_client.get_collection_info")
    def test_yellow_on_unknown_status(self, mock_info):
        mock_info.return_value = {"status": "indexing"}
        assert get_status() == "yellow"


# ---------------------------------------------------------------------------
# count_chunks_for_expert
# ---------------------------------------------------------------------------


class TestCountChunksForExpert:
    @patch("forge.core.qdrant_client._post")
    def test_success(self, mock_post):
        mock_post.return_value = _mock_response({"result": {"count": 55}})
        assert count_chunks_for_expert("alice") == 55

    @patch("forge.core.qdrant_client._post")
    def test_sends_correct_filter(self, mock_post):
        mock_post.return_value = _mock_response({"result": {"count": 0}})
        count_chunks_for_expert("bob")
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        must = body["filter"]["must"]
        assert must[0]["key"] == "creator"
        assert must[0]["match"]["value"] == "bob"

    @patch("forge.core.qdrant_client._post")
    def test_timeout_returns_zero(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("timed out")
        assert count_chunks_for_expert("alice") == 0

    @patch("forge.core.qdrant_client._post")
    def test_http_error_returns_zero(self, mock_post):
        mock_post.return_value = _mock_response({}, status_code=404)
        assert count_chunks_for_expert("alice") == 0

    @patch("forge.core.qdrant_client._post")
    def test_unexpected_error_returns_zero(self, mock_post):
        mock_post.side_effect = RuntimeError("boom")
        assert count_chunks_for_expert("alice") == 0


# ---------------------------------------------------------------------------
# get_all_expert_names
# ---------------------------------------------------------------------------


class TestGetAllExpertNames:
    @patch("forge.core.qdrant_client._post")
    def test_single_page(self, mock_post):
        mock_post.return_value = _mock_response({
            "result": {
                "points": [
                    {"payload": {"creator": "alice"}},
                    {"payload": {"creator": "bob"}},
                ],
                "next_page_offset": None,
            }
        })
        names = get_all_expert_names()
        assert names == ["alice", "bob"]

    @patch("forge.core.qdrant_client._post")
    def test_multi_page_pagination(self, mock_post):
        page1 = _mock_response({
            "result": {
                "points": [{"payload": {"creator": "alice"}}],
                "next_page_offset": 100,
            }
        })
        page2 = _mock_response({
            "result": {
                "points": [{"payload": {"creator": "bob"}}],
                "next_page_offset": None,
            }
        })
        mock_post.side_effect = [page1, page2]
        names = get_all_expert_names()
        assert names == ["alice", "bob"]

    @patch("forge.core.qdrant_client._post")
    def test_deduplication(self, mock_post):
        mock_post.return_value = _mock_response({
            "result": {
                "points": [
                    {"payload": {"creator": "alice"}},
                    {"payload": {"creator": "alice"}},
                    {"payload": {"creator": "bob"}},
                ],
                "next_page_offset": None,
            }
        })
        names = get_all_expert_names()
        assert names == ["alice", "bob"]

    @patch("forge.core.qdrant_client._post")
    def test_sorted_output(self, mock_post):
        mock_post.return_value = _mock_response({
            "result": {
                "points": [
                    {"payload": {"creator": "zara"}},
                    {"payload": {"creator": "alice"}},
                    {"payload": {"creator": "mike"}},
                ],
                "next_page_offset": None,
            }
        })
        names = get_all_expert_names()
        assert names == ["alice", "mike", "zara"]

    @patch("forge.core.qdrant_client._post")
    def test_empty_collection(self, mock_post):
        mock_post.return_value = _mock_response({
            "result": {"points": [], "next_page_offset": None}
        })
        assert get_all_expert_names() == []

    @patch("forge.core.qdrant_client._post")
    def test_missing_creator_skipped(self, mock_post):
        mock_post.return_value = _mock_response({
            "result": {
                "points": [
                    {"payload": {"creator": "alice"}},
                    {"payload": {}},
                    {"payload": None},
                    {},
                ],
                "next_page_offset": None,
            }
        })
        assert get_all_expert_names() == ["alice"]

    @patch("forge.core.qdrant_client._post")
    def test_timeout_returns_empty(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("timed out")
        assert get_all_expert_names() == []

    @patch("forge.core.qdrant_client._post")
    def test_http_error_returns_empty(self, mock_post):
        mock_post.return_value = _mock_response({}, status_code=500)
        assert get_all_expert_names() == []


# ---------------------------------------------------------------------------
# search_vectors
# ---------------------------------------------------------------------------


class TestSearchVectors:
    _VECTOR = [0.1] * 384

    @patch("forge.core.qdrant_client._post")
    def test_basic_search(self, mock_post):
        mock_post.return_value = _mock_response({
            "result": [
                {
                    "score": 0.95,
                    "payload": {
                        "creator": "alice",
                        "title": "Doc A",
                        "text": "hello",
                        "source": "web",
                        "chunk_index": 3,
                    },
                }
            ]
        })
        results = search_vectors(self._VECTOR, limit=5)
        assert len(results) == 1
        assert results[0]["score"] == 0.95
        assert results[0]["expert"] == "alice"
        assert results[0]["title"] == "Doc A"
        assert results[0]["text"] == "hello"
        assert results[0]["source"] == "web"
        assert results[0]["chunk_index"] == 3

    @patch("forge.core.qdrant_client._post")
    def test_expert_filter_added(self, mock_post):
        mock_post.return_value = _mock_response({"result": []})
        search_vectors(self._VECTOR, expert="bob")
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        must = body["filter"]["must"]
        assert any(f["key"] == "creator" and f["match"]["value"] == "bob" for f in must)

    @patch("forge.core.qdrant_client._post")
    def test_must_filters_passed(self, mock_post):
        mock_post.return_value = _mock_response({"result": []})
        custom = [{"key": "domain", "match": {"value": "security"}}]
        search_vectors(self._VECTOR, must_filters=custom)
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        must = body["filter"]["must"]
        assert any(f["key"] == "domain" for f in must)

    @patch("forge.core.qdrant_client._post")
    def test_should_filters_passed(self, mock_post):
        mock_post.return_value = _mock_response({"result": []})
        should = [{"key": "tag", "match": {"value": "important"}}]
        search_vectors(self._VECTOR, should_filters=should)
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert body["filter"]["should"][0]["key"] == "tag"

    @patch("forge.core.qdrant_client._post")
    def test_no_filter_when_none_provided(self, mock_post):
        mock_post.return_value = _mock_response({"result": []})
        search_vectors(self._VECTOR)
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "filter" not in body

    @patch("forge.core.qdrant_client._post")
    def test_combined_expert_and_must_filters(self, mock_post):
        mock_post.return_value = _mock_response({"result": []})
        custom = [{"key": "domain", "match": {"value": "security"}}]
        search_vectors(self._VECTOR, expert="alice", must_filters=custom)
        body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        must = body["filter"]["must"]
        assert len(must) == 2

    @patch("forge.core.qdrant_client._post")
    def test_empty_results(self, mock_post):
        mock_post.return_value = _mock_response({"result": []})
        assert search_vectors(self._VECTOR) == []

    @patch("forge.core.qdrant_client._post")
    def test_timeout_returns_empty(self, mock_post):
        mock_post.side_effect = httpx.TimeoutException("timed out")
        assert search_vectors(self._VECTOR) == []

    @patch("forge.core.qdrant_client._post")
    def test_http_error_returns_empty(self, mock_post):
        mock_post.return_value = _mock_response({}, status_code=500)
        assert search_vectors(self._VECTOR) == []

    @patch("forge.core.qdrant_client._post")
    def test_missing_payload_fields_default(self, mock_post):
        mock_post.return_value = _mock_response({
            "result": [{"score": 0.5, "payload": {}}]
        })
        results = search_vectors(self._VECTOR)
        assert results[0]["expert"] == ""
        assert results[0]["title"] == ""
        assert results[0]["text"] == ""
        assert results[0]["source"] == ""
        assert results[0]["chunk_index"] == 0

    @patch("forge.core.qdrant_client._post")
    def test_null_payload_handled(self, mock_post):
        mock_post.return_value = _mock_response({
            "result": [{"score": 0.3, "payload": None}]
        })
        results = search_vectors(self._VECTOR)
        assert len(results) == 1
        assert results[0]["expert"] == ""


# ---------------------------------------------------------------------------
# upsert_points
# ---------------------------------------------------------------------------


class TestUpsertPoints:
    def _make_point(self, pid: int) -> dict:
        return {"id": pid, "vector": [0.0] * 384, "payload": {"text": f"p{pid}"}}

    @patch("forge.core.qdrant_client._put")
    def test_single_batch(self, mock_put):
        mock_put.return_value = _mock_response({"status": "ok"})
        pts = [self._make_point(i) for i in range(5)]
        assert upsert_points(pts) == 5

    @patch("forge.core.qdrant_client._put")
    def test_empty_list_returns_zero(self, mock_put):
        assert upsert_points([]) == 0
        mock_put.assert_not_called()

    @patch("forge.core.qdrant_client._put")
    def test_batching_at_boundary(self, mock_put):
        mock_put.return_value = _mock_response({"status": "ok"})
        pts = [self._make_point(i) for i in range(_BATCH_SIZE)]
        assert upsert_points(pts) == _BATCH_SIZE
        assert mock_put.call_count == 1

    @patch("forge.core.qdrant_client._put")
    def test_batching_over_boundary(self, mock_put):
        mock_put.return_value = _mock_response({"status": "ok"})
        pts = [self._make_point(i) for i in range(_BATCH_SIZE + 1)]
        assert upsert_points(pts) == _BATCH_SIZE + 1
        assert mock_put.call_count == 2

    @patch("forge.core.qdrant_client._put")
    def test_multiple_full_batches(self, mock_put):
        mock_put.return_value = _mock_response({"status": "ok"})
        pts = [self._make_point(i) for i in range(_BATCH_SIZE * 3)]
        assert upsert_points(pts) == _BATCH_SIZE * 3
        assert mock_put.call_count == 3

    @patch("forge.core.qdrant_client._put")
    def test_partial_failure_counts_successes(self, mock_put):
        ok = _mock_response({"status": "ok"})
        err = _mock_response({}, status_code=500)
        mock_put.side_effect = [ok, err]
        pts = [self._make_point(i) for i in range(_BATCH_SIZE + 50)]
        result = upsert_points(pts)
        assert result == _BATCH_SIZE  # first batch ok, second fails

    @patch("forge.core.qdrant_client._put")
    def test_timeout_skips_batch(self, mock_put):
        ok = _mock_response({"status": "ok"})
        mock_put.side_effect = [httpx.TimeoutException("slow"), ok]
        pts = [self._make_point(i) for i in range(_BATCH_SIZE + 10)]
        result = upsert_points(pts)
        assert result == 10  # first batch times out, second (10 pts) ok

    @patch("forge.core.qdrant_client._put")
    def test_unexpected_error_skips_batch(self, mock_put):
        mock_put.side_effect = RuntimeError("unexpected")
        pts = [self._make_point(i) for i in range(5)]
        assert upsert_points(pts) == 0
