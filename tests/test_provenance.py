import hashlib
import json

import pytest

from uncertainty.utils.provenance import (
    build_provenance_record,
    collect_environment,
    hash_model_directory,
    sha256_file,
    write_provenance,
)


@pytest.fixture
def fake_model(tmp_path):
    root = tmp_path / "snapshots" / "abc123revision"
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"a": 1}')
    (root / "tokenizer.json").write_text("tok")
    (root / "model.safetensors").write_bytes(b"\x00\x01\x02" * 1000)
    (root / "README.md").write_text("ignored")
    return root


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    assert sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_hash_model_directory_selects_checkpoint_files_only(fake_model):
    files = hash_model_directory(fake_model)
    assert set(files) == {"config.json", "tokenizer.json", "model.safetensors"}
    assert files["model.safetensors"]["bytes"] == 3000


def test_hash_model_directory_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        hash_model_directory(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        hash_model_directory(tmp_path / "empty")


def test_record_has_revision_and_no_absolute_path(fake_model):
    record = build_provenance_record(fake_model)
    assert record["hf_snapshot_revision"] == "abc123revision"
    assert str(fake_model.parent) not in json.dumps(record)


def test_environment_lists_tracked_packages():
    env = collect_environment()
    assert env["python"] and "numpy" in env["packages"]


def test_write_provenance_creates_json(fake_model, tmp_path):
    out = tmp_path / "nested" / "prov.json"
    write_provenance(fake_model, out)
    assert json.loads(out.read_text())["model_name"] == "Qwen3-1.7B"
