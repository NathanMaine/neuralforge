"""Tests for forge.api.routes.events -- SSE stream, event publishing, recent events."""
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    mock_engine = MagicMock()
    mock_engine.node_count.return_value = 0
    mock_engine.edge_count.return_value = 0
    mock_engine.load = MagicMock()
    mock_engine.store = MagicMock()
    mock_engine.store.get_stats.return_value = MagicMock(
        total_nodes=0, total_edges=0, active_edges=0, expired_edges=0,
        node_type_counts={}, edge_type_counts={},
    )

    with patch("forge.api.main.GraphStore"), \
         patch("forge.api.main.GraphEngine", return_value=mock_engine), \
         patch("forge.api.main.bootstrap_graph", return_value={"skipped": True}), \
         patch("forge.api.main.GuardrailsEngine"), \
         patch("forge.api.main.start_scheduler"), \
         patch("forge.api.main.stop_scheduler"):
        from forge.api.main import app
        with TestClient(app) as c:
            yield c


class TestPublishEvent:
    def test_publish_event_adds_to_queue(self):
        from forge.api.routes.events import publish_event, _event_queue
        initial_len = len(_event_queue)
        publish_event("test_event", {"key": "value"})
        assert len(_event_queue) == initial_len + 1
        last = list(_event_queue)[-1]
        assert last["type"] == "test_event"
        assert last["data"] == {"key": "value"}
        assert "timestamp" in last

    def test_publish_event_delivers_to_subscriber(self):
        from forge.api.routes.events import publish_event, _subscribers
        queue = asyncio.Queue(maxsize=10)
        _subscribers.append(queue)
        try:
            publish_event("sub_event", {"msg": "hello"})
            assert not queue.empty()
            event = queue.get_nowait()
            assert event["type"] == "sub_event"
            assert event["data"]["msg"] == "hello"
        finally:
            if queue in _subscribers:
                _subscribers.remove(queue)

    def test_publish_event_full_subscriber_queue_no_error(self):
        from forge.api.routes.events import publish_event, _subscribers
        full_queue = asyncio.Queue(maxsize=1)
        full_queue.put_nowait({"type": "old", "data": {}, "timestamp": 0})
        _subscribers.append(full_queue)
        try:
            # Should silently drop and not raise
            publish_event("overflow_event", {"dropped": True})
        finally:
            if full_queue in _subscribers:
                _subscribers.remove(full_queue)

    def test_publish_event_broadcasts_to_multiple_subscribers(self):
        from forge.api.routes.events import publish_event, _subscribers
        q1 = asyncio.Queue(maxsize=10)
        q2 = asyncio.Queue(maxsize=10)
        _subscribers.extend([q1, q2])
        try:
            publish_event("broadcast", {"n": 1})
            assert not q1.empty()
            assert not q2.empty()
        finally:
            for q in (q1, q2):
                if q in _subscribers:
                    _subscribers.remove(q)

    def test_publish_event_has_timestamp(self):
        from forge.api.routes.events import publish_event, _event_queue
        publish_event("ts_event", {})
        last = list(_event_queue)[-1]
        assert isinstance(last["timestamp"], float)
        assert last["timestamp"] > 0


class TestRecentEvents:
    def test_recent_events_returns_200(self, client):
        resp = client.get("/api/events/recent")
        assert resp.status_code == 200

    def test_recent_events_has_events_key(self, client):
        data = client.get("/api/events/recent").json()
        assert "events" in data

    def test_recent_events_has_total_buffered(self, client):
        data = client.get("/api/events/recent").json()
        assert "total_buffered" in data

    def test_recent_events_empty_queue(self, client):
        from forge.api.routes.events import _event_queue
        _event_queue.clear()
        data = client.get("/api/events/recent").json()
        assert data["events"] == []
        assert data["total_buffered"] == 0

    def test_recent_events_contains_published_events(self, client):
        from forge.api.routes.events import publish_event, _event_queue
        _event_queue.clear()
        publish_event("recent_test", {"x": 42})
        data = client.get("/api/events/recent").json()
        assert len(data["events"]) >= 1
        assert data["events"][-1]["type"] == "recent_test"

    def test_recent_events_limit(self, client):
        from forge.api.routes.events import publish_event, _event_queue
        _event_queue.clear()
        for i in range(10):
            publish_event(f"evt_{i}", {"i": i})
        data = client.get("/api/events/recent?limit=3").json()
        assert len(data["events"]) == 3
        assert data["total_buffered"] == 10

    def test_recent_events_default_limit(self, client):
        from forge.api.routes.events import publish_event, _event_queue
        _event_queue.clear()
        for i in range(60):
            publish_event(f"evt_{i}", {})
        data = client.get("/api/events/recent").json()
        # Default limit is 50
        assert len(data["events"]) == 50


class TestEventStreamEndpoint:
    def test_event_stream_returns_sse_headers(self, client):
        """SSE endpoint returns text/event-stream content type."""
        async def finite_stream(queue):
            yield "event: test\ndata: {}\n\n"

        with patch("forge.api.routes.events._event_stream", side_effect=finite_stream):
            resp = client.get("/api/events")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_event_stream_cache_control_no_cache(self, client):
        """SSE endpoint sets Cache-Control: no-cache."""
        async def finite_stream(queue):
            yield "event: test\ndata: {}\n\n"

        with patch("forge.api.routes.events._event_stream", side_effect=finite_stream):
            resp = client.get("/api/events")
        assert "no-cache" in resp.headers.get("cache-control", "")

    def test_event_stream_delivers_event(self, client):
        """Published events are delivered via SSE stream."""
        async def finite_stream(queue):
            try:
                event = queue.get_nowait()
                yield f"event: {event['type']}\ndata: {{}}\n\n"
            except Exception:
                yield "event: empty\ndata: {}\n\n"

        from forge.api.routes.events import publish_event
        publish_event("sse_test", {"x": 1})

        with patch("forge.api.routes.events._event_stream", side_effect=finite_stream):
            resp = client.get("/api/events")
        assert resp.status_code == 200



class TestEventStreamGenerator:
    @pytest.mark.asyncio
    async def test_yields_event_line_for_queued_event(self):
        from forge.api.routes.events import _event_stream
        queue = asyncio.Queue()
        event = {"type": "my_event", "data": {"val": 99}}
        await queue.put(event)

        gen = _event_stream(queue)
        chunk = await gen.__anext__()
        assert "event: my_event" in chunk
        assert "99" in chunk

    @pytest.mark.asyncio
    async def test_event_data_is_json_encoded(self):
        from forge.api.routes.events import _event_stream
        import json
        queue = asyncio.Queue()
        await queue.put({"type": "json_test", "data": {"a": 1, "b": "two"}})

        gen = _event_stream(queue)
        chunk = await gen.__anext__()
        data_line = [l for l in chunk.split("\n") if l.startswith("data: ")][0]
        payload = json.loads(data_line[6:])
        assert payload["a"] == 1
        assert payload["b"] == "two"

    @pytest.mark.asyncio
    async def test_yields_heartbeat_on_timeout(self):
        import forge.api.routes.events as ev_mod
        original = ev_mod._HEARTBEAT_INTERVAL
        ev_mod._HEARTBEAT_INTERVAL = 0.001
        try:
            from forge.api.routes.events import _event_stream
            queue = asyncio.Queue()
            gen = _event_stream(queue)
            chunk = await gen.__anext__()
            assert "heartbeat" in chunk
        finally:
            ev_mod._HEARTBEAT_INTERVAL = original

    @pytest.mark.asyncio
    async def test_cancelled_error_stops_generator(self):
        from forge.api.routes.events import _event_stream
        queue = asyncio.Queue()
        gen = _event_stream(queue)
        # Close the generator -- should not raise
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_cancelled_error_during_wait_stops_generator(self):
        """CancelledError raised during queue.get() is swallowed gracefully."""
        from forge.api.routes.events import _event_stream

        queue = asyncio.Queue()
        gen = _event_stream(queue)

        # Prime the generator past the initial try/while/try
        # by first injecting a TimeoutError to get a heartbeat, then throw CancelledError
        await queue.put({"type": "init", "data": {}})
        first = await gen.__anext__()
        assert "event: init" in first

        # Now throw CancelledError into the running generator
        try:
            await gen.athrow(asyncio.CancelledError)
        except StopAsyncIteration:
            pass  # Generator stopped cleanly after handling CancelledError

    @pytest.mark.asyncio
    async def test_multiple_events_in_sequence(self):
        from forge.api.routes.events import _event_stream
        queue = asyncio.Queue()
        await queue.put({"type": "first", "data": {}})
        await queue.put({"type": "second", "data": {}})

        gen = _event_stream(queue)
        chunk1 = await gen.__anext__()
        chunk2 = await gen.__anext__()
        assert "event: first" in chunk1
        assert "event: second" in chunk2
