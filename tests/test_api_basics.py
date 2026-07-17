from api.main import _sse, graph_config, health


def test_health_endpoint_payload():
    # Call the route directly: this checks the liveness contract without
    # starting FastAPI's Postgres-dependent lifespan.
    assert health() == {"status": "ok"}


def test_sse_frames_json_data():
    assert _sse({"type": "progress", "message": "Searching…"}) == (
        'data: {"type": "progress", "message": "Searching\\u2026"}\n\n'
    )


def test_graph_config_adds_traceable_non_sensitive_metadata():
    config = graph_config("thread-123", "chat")

    assert config["configurable"] == {"thread_id": "thread-123"}
    assert config["run_name"] == "rahnuma-chat"
    assert config["tags"] == ["rahnuma", "chat"]
    assert config["metadata"] == {"thread_id": "thread-123", "api_route": "chat"}
