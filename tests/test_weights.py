"""weights.py and the node's --download-only and --no-download, without a GPU, the real weights or the internet.

Downloads go to a local HTTP server: URL files directly, Hugging Face files through HF_ENDPOINT (the node's own
request, its redirect to a second path, the token only to the first). Where network use must not happen, any use
fails the test. The node itself runs in this process up to its model, which a stand-in replaces. The part from
"the same in every node's test_weights.py" on is the same in every node."""

import contextlib
import hashlib
import http.server
import importlib
import io
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from xblick.session import open_session

from xblick_vjepa2 import node, weights

TERMS_LINE = "V-JEPA 2 is the work of Meta"  # the first line of weights.TERMS
NETWORK = True  # the weights come from the network
MODEL = ("xblick_vjepa2.model", "VJepa2")  # what loads the model, after the weights

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
    with pytest.raises(weights.WeightsError, match="model.safetensors.part: wrong SHA-256.*35aeab96"):
        weights.ensure([w], out=io.StringIO())
    assert not w.path.exists()


def test_download_only_with_a_directory_downloads_nothing(home, no_network, capsys):
    assert node.main(["--download-only", "--checkpoint", "/some/head/of/ours"]) == 0
    assert "no weights to download" in capsys.readouterr().out and not home.exists()


# ------------------------------------------------------------------------ the same in every node's test_weights.py

DATA = os.urandom(3 * 1024 * 1024 + 17)  # a stand-in weight file, larger than a few read chunks
DATA_SHA = hashlib.sha256(DATA).hexdigest()


class Server:
    """A local HTTP server of files: path -> bytes. ``/resolve/`` paths redirect to ``/cdn/`` (as Hugging Face's do);
    ``slow``: each MB after a pause; ``stall``: half the file, then nothing until the server stops. Records each GET
    and its Authorization header."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.gets: list[tuple[str, str | None]] = []
        self.slow = self.stall = False
        self.status: int | None = None  # answer every GET with this status
        self.stop = threading.Event()
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
                key = self.path.removeprefix("/cdn")
                if key not in owner.files:
                    self.send_error(404)
                    return
                data = owner.files[key]
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                with contextlib.suppress(OSError):
                    if owner.stall:
                        self.wfile.write(data[:len(data) // 2])
                        self.wfile.flush()
                        owner.stop.wait(60)
                        return
                    for i in range(0, len(data), 1 << 20):
                        if owner.slow:
                            time.sleep(0.3)
                        self.wfile.write(data[i:i + (1 << 20)])

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        if not self.stop.is_set():
            self.stop.set()
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


@pytest.fixture
def home(tmp_path, monkeypatch):
    """weights.WEIGHTS in tmp_path (the lock goes there too); no code from the setup required (stand_in may); a token
    that is no login of this PC."""
    root = tmp_path / "weights"
    monkeypatch.setattr(weights, "WEIGHTS", root)
    monkeypatch.setattr(weights, "REQUIRED", ())
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return root


def url_weight(root: Path, server: Server, name="a.bin", sha=DATA_SHA) -> weights.Weight:
    server.files[f"/{name}"] = DATA
    return weights.Weight(root / name, weights.Url(f"{server.url}/{name}"), len(DATA), sha)


def place(files: dict[Path, bytes]):
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


# ------------------------------------------------------------------------------------------------ all present


def test_all_present_touches_neither_network_nor_disk(home, stand_in, no_network):
    place(stand_in)
    before = sorted(home.rglob("*"))
    out = io.StringIO()
    assert weights.ensure(default_weights(), out=out) == [] and out.getvalue() == ""
    assert weights.ensure(default_weights(), download=False) == []
    assert sorted(home.rglob("*")) == before  # no lock file either


def test_download_only_with_all_present_prints_one_line(home, stand_in, no_network, capsys):
    place(stand_in)
    assert node.main(["--download-only"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and "nothing to download" in lines[0]


# ---------------------------------------------------------------------------------------------------- downloads


def test_a_hugging_face_file_is_downloaded_checked_and_renamed(home, server, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", server.url)
    server.files["/model.bin"] = DATA
    w = weights.Weight(home / "sub" / "model.bin", weights.HuggingFace("org/model", "abc123", "model.bin"), len(DATA))
    out = io.StringIO()
    assert weights.ensure([w], out=out) == [w]
    assert w.path.read_bytes() == DATA and not list(home.rglob("*.part"))
    # the node's URL; the token to Hugging Face, not to the host it redirects to
    assert server.gets == [("/org/model/resolve/abc123/model.bin", "Bearer secret"), ("/cdn/model.bin", None)]
    text = out.getvalue()
    assert "to download: " in text and TERMS_LINE in text and "Hugging Face org/model at revision abc123" in text
    assert text.index(TERMS_LINE) < text.index("from Hugging Face")  # the terms before the download


def test_a_url_file_is_checked_against_its_sha256(home, server):
    w = url_weight(home, server)
    assert weights.ensure([w], out=io.StringIO()) == [w]
    assert w.path.read_bytes() == DATA
    assert weights.ensure([w], out=io.StringIO()) == [] and len(server.gets) == 1


def test_a_wrong_checksum_keeps_the_download_aside_and_says_to_remove_it(home, server):
    w = url_weight(home, server, sha="0" * 64)
    with pytest.raises(weights.WeightsError) as error:
        weights.ensure([w], out=io.StringIO())
    assert "a.bin.part: wrong SHA-256" in str(error.value) and "remove it" in str(error.value)
    assert not w.path.exists() and (home / "a.bin.part").read_bytes() == DATA


def test_a_download_of_another_length_is_refused_and_kept_aside(home, server):
    w = url_weight(home, server, sha=None)
    w = weights.Weight(w.path, w.source, len(DATA) + 1)  # a size in code that the source no longer has
    with pytest.raises(weights.WeightsError, match=rf"a.bin.part: {len(DATA)} bytes, expected {len(DATA) + 1}.*remove"):
        weights.ensure([w], out=io.StringIO())
    assert not w.path.exists() and (home / "a.bin.part").exists()


def test_every_file_of_the_node_has_its_exact_size_and_sha256():
    for w in default_weights():
        assert w.size > 0 and (w.sha256 is None) == (w.path.name in WITHOUT_SHA256), w.path.name
        assert w.sha256 is None or len(w.sha256) == 64


def test_download_only_checks_present_files_against_their_sha256(home, no_network):
    w = weights.Weight(home / "a.bin", weights.Url("http://127.0.0.1:9/a.bin"), 1, DATA_SHA)
    place({w.path: b"x"})
    with pytest.raises(weights.WeightsError, match="a.bin: wrong SHA-256.*remove it"):
        weights.ensure([w], verify=True)
    assert w.path.exists()  # not deleted


def test_a_gated_repository_says_how_to_get_access(home, server, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", server.url)
    gated = weights.Weight(home / "c.pt", weights.HuggingFace("facebook/sam3.1", "main", "c.pt", gated=True), 1)
    server.status = 401
    with pytest.raises(weights.WeightsError) as error:
        weights.ensure([gated], out=io.StringIO())
    message = str(error.value)
    assert "https://huggingface.co/facebook/sam3.1" in message and "hf auth login" in message and "HF_TOKEN" in message


@pytest.mark.skipif(not NETWORK, reason="this node's weights come from no network")
def test_without_a_network_download_only_ends_with_one_line(home, server, stand_in, capsys):
    server.close()  # no server any more: as without a network
    assert node.main(["--download-only"]) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "cannot download from 127.0.0.1" in err[0] and "connect this PC" in err[0]
    assert all(not path.exists() for path in stand_in) and not list(home.rglob("*.part"))


def test_missing_code_from_the_setup_ends_with_one_line(home, stand_in, monkeypatch, no_network, capsys):
    monkeypatch.setattr(weights, "REQUIRED", ((home.parent / "third_party" / "code", "the upstream code"),))
    with pytest.raises(weights.WeightsError, match=r"code is missing \(the upstream code\): run tools/setup.sh"):
        weights.ensure(default_weights(), out=io.StringIO())
    assert node.main(["--download-only"]) == 1 and "run tools/setup.sh" in capsys.readouterr().err
    assert not home.exists()  # nothing written: no lock, no directory


def test_hf_hub_offline_is_kept(home, monkeypatch, no_network):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    w = weights.Weight(home / "m.bin", weights.HuggingFace("org/model", "abc123", "m.bin"), 1)
    with pytest.raises(weights.WeightsError, match="HF_HUB_OFFLINE"):
        weights.ensure([w], out=io.StringIO())


def test_progress_in_a_log_file_has_lines_and_no_carriage_returns(home, server):
    w = url_weight(home, server)
    log = io.StringIO()  # no terminal
    weights.ensure([w], out=log)
    progress = [line for line in log.getvalue().splitlines() if "a.bin: " in line and "%" in line]
    assert "\r" not in log.getvalue() and 3 <= len(progress) <= 11 and "100%" in progress[-1]

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    w.path.unlink()
    terminal = Terminal()
    weights.ensure([w], out=terminal)
    assert "\r" in terminal.getvalue()


# ----------------------------------------------------- interrupts and two processes at once (own processes)

CHILD = f"""
import signal
import sys
from pathlib import Path
from {weights.__name__} import Url, Weight, WeightsError, ensure, fail
import {weights.__name__} as weights
signal.signal(signal.SIGINT, signal.default_int_handler)  # also where a background shell ignores it
root, url, sha, size = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
weights.WEIGHTS = root
weights.REQUIRED = ()
try:
    ensure([Weight(root / "a.bin", Url(url), size, sha)])
except WeightsError as error:
    sys.exit(fail(error))
print("done", flush=True)
"""


def child(root: Path, server: Server) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", CHILD, str(root), f"{server.url}/a.bin", DATA_SHA, str(len(DATA))],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_for(condition, seconds=20.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.05)


@pytest.mark.parametrize(("sig", "code"), [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_an_interrupted_download_leaves_no_file(home, server, sig, code):
    server.files["/a.bin"] = DATA
    server.stall = True
    process = child(home, server)
    try:
        wait_for(lambda: (home / "a.bin.part").exists() and (home / "a.bin.part").stat().st_size > 0)
        process.send_signal(sig)
        _, err = process.communicate(timeout=10)  # promptly: well before the server's stall ends
    finally:
        process.kill()
    assert process.returncode == code and "stopped (" in err and "Traceback" not in err
    assert not (home / "a.bin").exists() and not (home / "a.bin.part").exists()


def test_two_processes_at_once_download_once(home, server):
    server.files["/a.bin"] = DATA
    server.slow = True  # about a second: the second process starts while the first downloads
    first = child(home, server)
    wait_for(lambda: (home / "a.bin.part").exists())
    second = child(home, server)
    outputs = [p.communicate(timeout=60) for p in (first, second)]
    assert [p.returncode for p in (first, second)] == [0, 0], outputs
    assert [path for path, _ in server.gets] == ["/a.bin"]
    assert "another process downloaded the weights" in outputs[1][0]
    assert (home / "a.bin").read_bytes() == DATA


# ------------------------------------------------------------------------------------------- the command line


def test_download_only_needs_no_start_options_and_excludes_no_download(capsys):
    with pytest.raises(SystemExit) as exit_:
        node.main(["--download-only", "--no-download"])
    assert exit_.value.code == 2 and "not allowed with" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exit_:
        node.main(["--no-download"])  # a start needs its required options
    assert exit_.value.code == 2 and "required" in capsys.readouterr().err


def test_download_only_downloads_and_exits_0(home, stand_in, capsys):
    assert node.main(["--download-only"]) == 0
    assert all(path.read_bytes() == data for path, data in stand_in.items())
    assert f"weights complete: {len(stand_in)} downloaded" in capsys.readouterr().out.splitlines()[-1]


class Started(Exception):
    """The node got past its weights to the model."""


@pytest.fixture
def started(monkeypatch, tmp_path):
    """The node's model replaced by one that raises Started; a Zenoh session the node connects to; the node's
    parameters kept."""
    from xblick import node as core_node

    module, name = MODEL

    def stand_in_model(*args, **kwargs):
        raise Started

    monkeypatch.setattr(importlib.import_module(module), name, stand_in_model)
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
    for closable in (*kept, *sessions):  # a start that got to the model left them open (the model raised)
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
