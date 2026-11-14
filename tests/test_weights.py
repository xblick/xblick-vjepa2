"""The node's weights (weights.py) and its --download-only and --no-download, without a GPU, the real weights or the
internet: the files, their checksums and sources, and the node's start up to the end of its weights step.

The download itself is the core's xblick.weights, which tests it in its own tests/test_weights.py. Downloads here go
to a local HTTP server: URL files directly, Hugging Face files through HF_ENDPOINT. Where network use must not
happen, any use fails the test. The part from "the same in every node's test_weights.py" on is the same in every
node."""

import contextlib
import hashlib
import http.server
import io
import os
import signal
import socket
import threading
from pathlib import Path

import pytest
from xblick import weights as xblick_weights
from xblick.session import open_session

from xblick_vjepa2 import node, weights

TERMS_LINE = "V-JEPA 2 is the work of Meta"  # the first line of weights.TERMS
NETWORK = True  # the weights come from the network

WITHOUT_SHA256 = ()  # files whose source states no SHA-256


def start_options(tmp_path) -> list[str]:
    """A start's required options."""
    return ["--camera", "test/camera"]


def default_weights():
    return weights.needed("ssv2")


@pytest.fixture
def stand_in(home, server, monkeypatch):
    """The default checkpoint's files as stand-ins on the local server, through HF_ENDPOINT: {path: content}."""
    monkeypatch.setenv("HF_ENDPOINT", server.url)
    server.files["/config.json"] = b"{}"
    server.files["/model.safetensors"] = DATA
    repo, revision, _ = weights.CHECKPOINTS["ssv2"]
    monkeypatch.setitem(weights.CHECKPOINTS, "ssv2", (repo, revision, {
        "config.json": (2, hashlib.sha256(b"{}").hexdigest()), "model.safetensors": (len(DATA), DATA_SHA)}))
    return {home / "ssv2" / "config.json": b"{}", home / "ssv2" / "model.safetensors": DATA}


def test_the_files_are_the_ssv2_checkpoint_at_its_revision():
    ws = weights.needed("ssv2")
    assert [w.path for w in ws] == [weights.WEIGHTS / "ssv2" / "config.json",
                                    weights.WEIGHTS / "ssv2" / "model.safetensors"]
    for w in ws:
        assert w.source.repo == "facebook/vjepa2-vitl-fpc16-256-ssv2"
        assert w.source.revision == "4aa02df83918538fc21cfaf576382fa20e489a80" and not w.source.gated
        assert w.source.url().endswith(f"/facebook/vjepa2-vitl-fpc16-256-ssv2/resolve/{w.source.revision}/"
                                       + w.path.name)
    assert [(w.size, w.sha256) for w in ws] == [  # Hugging Face's lfs.oid; config.json hashed at the revision
        (14_953, "9ea266370f726ba85ba03080abab9632233d0742a084f0106d90d909772e4aab"),
        (1_502_018_592, "35aeab96e8513d9cc60ee08c1642dd121c86452c926ce3599b3317896a320396")]
    assert weights.needed("/some/head/of/ours") == []  # a directory of one's own: nothing to download


def test_a_changed_checkpoint_file_is_refused(home, server, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", server.url)
    server.files["/model.safetensors"] = DATA
    real = weights.needed("ssv2")[1]
    w = weights.Weight(home / "model.safetensors", real.source, len(DATA), real.sha256)  # the real SHA-256
    with pytest.raises(xblick_weights.WeightsError, match="model.safetensors.part: wrong SHA-256.*35aeab96"):
        weights.STORE.ensure([w], out=io.StringIO())
    assert not w.path.exists()


def test_download_only_with_a_directory_downloads_nothing(home, no_network, capsys):
    assert node.main(["--download-only", "--checkpoint", "/some/head/of/ours"]) == 0
    assert "no weights to download" in capsys.readouterr().out and not home.exists()


# ------------------------------------------------------------------------ the same in every node's test_weights.py

DATA = os.urandom(3 * 1024 * 1024 + 17)  # a stand-in weight file, larger than a few read chunks
DATA_SHA = hashlib.sha256(DATA).hexdigest()


class Server:
    """A local HTTP server of files: path -> bytes. ``/resolve/`` paths redirect to ``/cdn/`` (as Hugging Face's do);
    ``status``: every GET answered with it. Records each GET and its Authorization header."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.gets: list[tuple[str, str | None]] = []
        self.status: int | None = None
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                owner.gets.append((self.path, self.headers.get("Authorization")))
                if owner.status:
                    self.send_error(owner.status)
                    return
                if "/resolve/" in self.path:
                    self.send_response(302)
                    self.send_header("Location", "/cdn/" + self.path.rsplit("/", 1)[1])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = owner.files.get(self.path.removeprefix("/cdn"))
                if data is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                with contextlib.suppress(OSError):
                    self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


@pytest.fixture
def no_network(monkeypatch):
    """Any network use of this process fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"network use: {args}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def use_store(monkeypatch, required=()):
    """The node's STORE as it is, with other required paths (code from the setup)."""
    store = weights.STORE
    monkeypatch.setattr(weights, "STORE", xblick_weights.Store(store.prog, store.root, store.terms, required,
                                                               repo=store.repo))


@pytest.fixture
def home(tmp_path, monkeypatch):
    """weights.WEIGHTS and the STORE's root (the lock goes there) in tmp_path; no code from the setup required
    (stand_in may); a token that is no login of this PC."""
    root = tmp_path / "weights"
    monkeypatch.setattr(weights, "WEIGHTS", root)
    monkeypatch.setattr(weights, "STORE", xblick_weights.Store(weights.PROG, root, weights.TERMS, (),
                                                               repo=weights.REPO))
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return root


def place(files: dict[Path, bytes]):
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_the_store_is_the_nodes():
    store = weights.STORE
    assert (store.prog, store.root, store.terms, store.required, store.repo) == (
        weights.PROG, weights.WEIGHTS, weights.TERMS, tuple(weights.REQUIRED), weights.REPO)
    assert store.prog == node.main.__module__.split(".")[0].replace("_", "-")


def test_every_file_of_the_node_has_its_exact_size_and_sha256():
    for w in default_weights():
        assert isinstance(w, xblick_weights.Weight) and w.size > 0, w.path.name
        assert (w.sha256 is None) == (w.path.name in WITHOUT_SHA256), w.path.name
        assert w.sha256 is None or len(w.sha256) == 64


def test_all_present_touches_neither_network_nor_disk(home, stand_in, no_network):
    place(stand_in)
    before = sorted(home.rglob("*"))
    out = io.StringIO()
    assert weights.STORE.ensure(default_weights(), out=out) == [] and out.getvalue() == ""
    assert sorted(home.rglob("*")) == before  # no lock file either


def test_download_only_with_all_present_prints_one_line(home, stand_in, no_network, capsys):
    place(stand_in)
    assert node.main(["--download-only"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and "nothing to download" in lines[0]


def test_download_only_downloads_with_the_terms_and_exits_0(home, stand_in, capsys):
    assert node.main(["--download-only"]) == 0
    assert all(path.read_bytes() == data for path, data in stand_in.items())
    out = capsys.readouterr().out
    assert TERMS_LINE in out and f"weights complete: {len(stand_in)} downloaded" in out.splitlines()[-1]
    assert not list(home.rglob("*.part"))


def test_download_only_needs_no_start_options_and_excludes_no_download(capsys):
    with pytest.raises(SystemExit) as exit_:
        node.main(["--download-only", "--no-download"])
    assert exit_.value.code == 2 and "not allowed with" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_:
        node.main(["--no-download"])  # a start needs its required options
    assert exit_.value.code == 2 and "required" in capsys.readouterr().err


class Started(Exception):
    """The node got past its weights: what follows (the model, the GPU) is not this test's."""


@pytest.fixture
def started(monkeypatch, tmp_path):
    """The STORE's ensure raising Started once it returned (the node's start stops there, before anything touches
    the GPU: the tests need none); a Zenoh session the node connects to; the node's parameters kept."""
    from xblick import node as core_node

    ensure = weights.STORE.ensure

    def ensure_then_stop(*args, **kwargs):
        ensure(*args, **kwargs)
        raise Started

    monkeypatch.setattr(weights.STORE, "ensure", ensure_then_stop)
    kept = []

    class Kept(core_node.Parameters):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            kept.append(self)

    monkeypatch.setattr(core_node, "Parameters", Kept)
    sessions = []
    opened = node.open_from_args

    def keep_session(args):
        sessions.append(opened(args))
        return sessions[-1]

    monkeypatch.setattr(node, "open_from_args", keep_session)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    session = open_session(listen=[f"127.0.0.1:{port}"])
    sigterm = signal.getsignal(signal.SIGTERM)  # node.start sets its own
    yield [*start_options(tmp_path), "--connect", f"127.0.0.1:{port}"], kept
    for closable in (*kept, *sessions):  # a start that got past the weights left them open (Started)
        with contextlib.suppress(Exception):  # closed already by a node that ended
            closable.close()
    signal.signal(signal.SIGTERM, sigterm)
    session.close()


def test_no_download_ends_at_once_naming_the_missing_files(home, server, stand_in, started, capsys):
    start, kept = started
    assert node.main([*start, "--no-download"]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and all(path.name in err[0] for path in stand_in) and "--download-only" in err[0]
    assert server.gets == [] and not any(path.exists() for path in stand_in)
    # a read-only parameter, as every start option; --download-only is a mode of the command line, no parameter
    assert kept[0].descriptor("no_download").read_only and kept[0]["no_download"] is True
    assert "download_only" not in kept[0].names()


def test_no_download_with_all_present_starts_as_usual(home, stand_in, started, no_network):
    start, _ = started
    place(stand_in)
    with pytest.raises(Started):
        node.main([*start, "--no-download"])


def test_a_start_downloads_what_is_missing_first(home, stand_in, started):
    start, _ = started
    with pytest.raises(Started):
        node.main(start)
    assert all(path.read_bytes() == data for path, data in stand_in.items())
