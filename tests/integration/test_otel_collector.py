"""D2's acceptance bar: a REAL OpenTelemetry Collector accepts what we emit.

A unit round-trip proves only self-consistency. This plan's own review found
the wrong top-level message named in the design (`ResourceMetrics` instead of
`ExportMetricsServiceRequest`), and a same-library round-trip would have passed
on it - a collector is the only thing that can tell us the wire shape is right.

Gated on a real collector being available, the way
``test_container_image.py`` gates on a Docker daemon. Two ways to get one, both
the genuine article:

* ``docker run otel/opentelemetry-collector`` - the CI path (``container.yml``).
* an ``otelcol`` / ``otelcol-contrib`` binary on PATH - a developer or sandbox
  without a Docker daemon. Set ``BOBI_TEST_OTELCOL`` to point at one explicitly.

The test body is identical either way: it POSTs through the real exporter and
asserts on what the collector's ``debug`` exporter actually RENDERED, never on
a local encode.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from .conftest import _free_port

REPO_ROOT = Path(__file__).resolve().parents[2]

# Brings up a container - excluded from integration-fast via `-m "not docker"`
# so it never runs on every PR, and homed in container.yml with the other
# docker-marked suites.
pytestmark = pytest.mark.docker

COLLECTOR_IMAGE = "otel/opentelemetry-collector:0.144.0"

CONFIG = """\
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:{port}

exporters:
  debug:
    verbosity: detailed

service:
  telemetry:
    logs:
      level: info
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [debug]
    logs:
      receivers: [otlp]
      exporters: [debug]
"""


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=15)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def _otelcol_binary() -> str | None:
    explicit = os.environ.get("BOBI_TEST_OTELCOL")
    if explicit and Path(explicit).is_file() and os.access(explicit, os.X_OK):
        return explicit
    return shutil.which("otelcol") or shutil.which("otelcol-contrib")


def _collector_available() -> bool:
    return _docker_ok() or _otelcol_binary() is not None


requires_collector = pytest.mark.skipif(
    not _collector_available(),
    reason="no OpenTelemetry Collector available (need a docker daemon or an "
           "otelcol binary on PATH / BOBI_TEST_OTELCOL)",
)


class _Collector:
    """A running collector, however it was launched, plus its rendered output."""

    def __init__(self, proc: subprocess.Popen, port: int, log: Path,
                 container: str | None):
        self.proc = proc
        self.port = port
        self._log = log
        self._container = container

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def output(self) -> str:
        if self._container:
            done = subprocess.run(
                ["docker", "logs", self._container],
                capture_output=True, text=True, timeout=30,
            )
            return done.stdout + done.stderr
        return self._log.read_text(errors="replace")

    def await_output(self, needle: str, timeout: float = 20.0) -> str:
        """The collector renders asynchronously, so poll its output."""
        deadline = time.monotonic() + timeout
        text = ""
        while time.monotonic() < deadline:
            text = self.output()
            if needle in text:
                return text
            time.sleep(0.2)
        raise AssertionError(
            f"collector never rendered {needle!r}. Output was:\n{text[-4000:]}"
        )

    def stop(self) -> None:
        if self._container:
            subprocess.run(["docker", "rm", "-f", self._container],
                           capture_output=True, timeout=60)
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _await_ready(endpoint: str, proc: subprocess.Popen, timeout: float = 60.0) -> None:
    """Ready = the OTLP receiver ANSWERS. A TCP connect proves the wrong thing.

    Under Docker's userland proxy, ``docker-proxy`` binds and accepts on the
    published host port as soon as the container is created, before otelcol
    binds 4318 inside it. A ``connect_ex`` probe therefore succeeds while the
    receiver is still down, the proxy's dial into the container fails, and the
    POST the test has already written comes back as ``[Errno 104] Connection
    reset by peer``.

    Any HTTP status answers the only question that matters, so the probe does
    not check one. The body is empty, which is a valid empty
    ``ExportMetricsServiceRequest``: the debug exporter renders nothing for it,
    so the probe cannot pollute the pid-keyed marker assertions.
    """
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"collector exited early with {proc.returncode}")
        try:
            httpx.post(f"{endpoint}/v1/metrics", content=b"",
                       headers={"Content-Type": "application/x-protobuf"},
                       timeout=2.0)
            return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise AssertionError(f"collector never answered on {endpoint} within {timeout}s")


@pytest.fixture(scope="module")
def collector(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("otelcol")
    port = _free_port()
    log = tmp / "collector.log"
    binary = _otelcol_binary()

    if binary:
        config = tmp / "config.yaml"
        config.write_text(CONFIG.format(port=port))
        handle = log.open("w")
        proc = subprocess.Popen(
            [binary, "--config", str(config)], stdout=handle, stderr=subprocess.STDOUT,
        )
        container = None
    else:
        config = tmp / "config.yaml"
        # The collector runs as a non-root uid in the image and only reads this.
        config.write_text(CONFIG.format(port=4318))
        config.chmod(0o644)
        container = f"bobi-otel-test-{port}"
        # Same log file as the binary path. DEVNULL here made a failed image
        # pull or a rejected config invisible: the test could only report that
        # nothing ever answered, never why.
        handle = log.open("w")
        proc = subprocess.Popen(
            ["docker", "run", "--rm", "--name", container,
             "-p", f"127.0.0.1:{port}:4318",
             "-v", f"{config}:/etc/otelcol/config.yaml:ro",
             COLLECTOR_IMAGE, "--config", "/etc/otelcol/config.yaml"],
            stdout=handle, stderr=subprocess.STDOUT,
        )

    running = _Collector(proc, port, log, container)
    try:
        _await_ready(running.endpoint, proc)
        yield running
    finally:
        running.stop()


@pytest.fixture
def exporting(collector, tmp_path, monkeypatch):
    """Point the real exporter at the running collector."""
    from bobi.otel import config as otel_config

    for name in list(os.environ):
        if name.startswith("OTEL_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint)
    return otel_config


@requires_collector
def test_collector_accepts_metric(
    collector, exporting, tmp_path
):
    from bobi.otel.export import MetricSpec, export_metric

    marker = f"bobi.collector.probe.{os.getpid()}"
    resource = {
        "service.name": "bobi",
        "service.instance.id": "eng-team",
        "bobi.fleet": "moda",
        "bobi.platform": "docker",
        "bobi.agent": "eng-team",
        "bobi.team": "moda-eng-team",
    }
    cfg = exporting.resolve_config(tmp_path, "metrics")
    assert cfg.url == f"{collector.endpoint}/v1/metrics"

    # The real code path: build the envelope, POST it, enforce the success
    # contract. A rejection at HTTP 200 would raise here.
    export_metric(cfg, MetricSpec(name=marker, value=42, unit="1",
                                  description="collector probe",
                                  attributes={"queue": "inbox"}), resource)

    rendered = collector.await_output(marker)
    # What the COLLECTOR saw, not what we encoded.
    assert "-> service.name: Str(bobi)" in rendered
    assert "-> bobi.fleet: Str(moda)" in rendered
    assert "-> bobi.agent: Str(eng-team)" in rendered
    assert "-> bobi.team: Str(moda-eng-team)" in rendered
    assert "-> queue: Str(inbox)" in rendered
    assert "Value: 42" in rendered
    assert re.search(r"DataType:\s*Sum", rendered)
    assert re.search(r"IsMonotonic:\s*true", rendered)
    assert "AggregationTemporality: Delta" in rendered
    assert "Descriptor Name: " in rendered or marker in rendered


@requires_collector
def test_collector_accepts_log(collector, exporting, tmp_path):
    from bobi.otel.export import LogSpec, export_log

    marker = f"collector log probe {os.getpid()}"
    resource = {"service.name": "bobi", "bobi.fleet": "moda"}
    cfg = exporting.resolve_config(tmp_path, "logs")
    assert cfg.url == f"{collector.endpoint}/v1/logs"

    export_log(cfg, LogSpec(body=marker, severity="error",
                            attributes={"stage": "reconcile"}), resource)

    rendered = collector.await_output(marker)
    assert f"Body: Str({marker})" in rendered
    assert "SeverityText: ERROR" in rendered
    assert "SeverityNumber: Error(17)" in rendered
    assert "-> stage: Str(reconcile)" in rendered
    assert "-> bobi.fleet: Str(moda)" in rendered


@requires_collector
def test_collector_rejects_a_bare_resource_metrics_body(collector, exporting, tmp_path):
    """The trap, proven against the real thing.

    A body serialized as the INNER `ResourceMetrics` message is what a naive
    reading of the OTLP protos produces, and a same-type round-trip test
    accepts it. The collector does not.
    """
    import httpx
    from opentelemetry.proto.metrics.v1.metrics_pb2 import ResourceMetrics
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

    wrong = ResourceMetrics(
        resource=Resource(attributes=[
            KeyValue(key="service.name", value=AnyValue(string_value="bobi"))
        ])
    )
    response = httpx.post(
        f"{collector.endpoint}/v1/metrics",
        content=wrong.SerializeToString(),
        headers={"Content-Type": "application/x-protobuf"},
        timeout=10,
    )
    assert response.status_code == 400, (
        f"expected the collector to reject the inner message, got "
        f"{response.status_code}: {response.text[:400]}"
    )


@requires_collector
def test_the_cli_reaches_the_collector_end_to_end(collector, bobi_install, monkeypatch):
    """`bobi agent <name> otel check --send` against a real collector.

    Runs in a subprocess with the fixture's isolated BOBI_HOME so it exercises
    argument parsing, the agent-group binding, and identity resolution the way
    an agent actually invokes it.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    env.update({
        "BOBI_HOME": str(bobi_install.agents_dir.parent),
        "OTEL_EXPORTER_OTLP_ENDPOINT": collector.endpoint,
        "BOBI_FLEET": "moda",
        "BOBI_INSTANCE": "eng-team",
        "PYTHONPATH": str(REPO_ROOT),
    })
    env.pop("BOBI_ROOT", None)

    done = subprocess.run(
        [sys.executable, "-m", "bobi.cli", "agent", bobi_install.agent_name,
         "otel", "check", "--send"],
        capture_output=True, text=True, timeout=120, cwd=REPO_ROOT, env=env,
    )
    assert done.returncode == 0, f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    assert "sent bobi.otel.check" in done.stdout

    rendered = collector.await_output("bobi.otel.check")
    assert "-> bobi.fleet: Str(moda)" in rendered
    assert "-> service.instance.id: Str(eng-team)" in rendered
    assert f"-> bobi.agent: Str({bobi_install.agent_name})" in rendered
    assert re.search(r"DataType:\s*Gauge", rendered)
