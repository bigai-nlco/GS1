import asyncio
import json
import threading
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from gs1.api.app import create_app
from gs1.api.local import BackendError, LocalBackend
from gs1.api.settings import Settings


def test_model_server_handles_all_primitives(checkpoint, decision_request):
    settings = Settings(artifact=checkpoint, device="cpu", precision="fp32")
    with TestClient(create_app(settings)) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert isinstance(models.json()["models"][0]["release_date"], str)
        assert client.get("/docs").status_code == 200
        response = client.post("/v1/systemone", json={"model": "GS1", **decision_request})
        assert response.status_code == 200, response.text
        assert set(response.json()["answers"]) == set(decision_request["questions"])
        assert response.json()["usage"]["output_tokens"] == 0
        assert client.post("/v1/systemone", json={"model": "unknown", **decision_request}).status_code == 422


def test_server_rejects_invalid_primitive(checkpoint, decision_request):
    settings = Settings(artifact=checkpoint, device="cpu", precision="fp32")
    decision_request["questions"]["score"]["criteria"] = ["single"]
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/systemone", json={"model": "GS1", **decision_request}).status_code == 422


def test_official_typesafe_sdk_uses_local_api(checkpoint, decision_request):
    import httpx2
    from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

    with TestClient(create_app(Settings(artifact=checkpoint, device="cpu", precision="fp32"))) as server:
        paths = []

        def forward(request):
            paths.append(request.url.path)
            response = server.request(
                request.method, request.url.path, content=request.content, headers=dict(request.headers)
            )
            return httpx2.Response(response.status_code, content=response.content, headers=response.headers)

        with TypeSafeClient(
            base_url="http://gs1.local",
            api_key="local",
            model="GS1",
            retry=RetryPolicy(max_retries=0),
            transport=httpx2.MockTransport(forward),
        ) as client:
            models = client.models.list()
            result = client.system_one(
                state=decision_request["state"],
                questions={
                    "choice": Choice(instructions="Choose.", criteria={"a": "Cost 2", "b": "Cost 12"}),
                    "noul": Noul(instructions="Is battery above reserve?"),
                    "score": Score(instructions="Rate battery.", criteria=["Low", "Medium", "High"]),
                },
            )
        assert paths == ["/v1/models", "/v1/systemone"]
        assert models.models[0].name == "GS1"
        assert result.choices["choice"].choice in {"a", "b"}
        assert sum(result.choices["choice"].probabilities.values()) == pytest.approx(1)
        assert 0 <= result.nouls["noul"].noul <= 1
        assert 0 <= result.scores["score"].score <= 2
        assert result.usage.output_tokens == 0


def test_replica_command_resolves_string_model_path(checkpoint, tmp_path, capsys):
    from scripts.serve_replicas import main

    with patch("scripts.serve_replicas.devices", return_value=["GPU-test"]):
        main(["--model", str(checkpoint), "--runs-dir", str(tmp_path / "replicas"), "--dry-run"])
    replicas = json.loads(capsys.readouterr().out)
    command = replicas[0]["command"]
    assert command[command.index("--model-path") + 1] == str(checkpoint.resolve())


@pytest.mark.asyncio
async def test_deadline_keeps_execution_owned_until_shutdown(checkpoint, decision_request):
    backend = LocalBackend(
        Settings(artifact=checkpoint, device="cpu", precision="fp32", deadline_seconds=0.2)
    )
    await backend.start()
    entered, release = threading.Event(), threading.Event()
    evaluate = backend.engines[0].evaluate

    def slow(*args, **kwargs):
        entered.set()
        release.wait(5)
        return evaluate(*args, **kwargs)

    backend.engines[0].evaluate = slow
    closing = None
    try:
        pending = asyncio.create_task(backend.evaluate(decision_request))
        assert await asyncio.to_thread(entered.wait, 2)
        with pytest.raises(BackendError) as error:
            await pending
        assert error.value.status == 504
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing or backend.close(), 5)
