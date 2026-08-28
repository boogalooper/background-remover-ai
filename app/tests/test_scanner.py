from pathlib import Path

from app.core.scanner import common_source_root, scan_inputs, suggested_output_dir


def test_scan_inputs_recursive_and_excludes_output(tmp_path: Path):
    src = tmp_path / "photos"
    out = src / "Background Removed"
    sub = src / "classA"
    sub.mkdir(parents=True)
    out.mkdir()
    (src / "a.JPG").write_bytes(b"x")
    (sub / "b.png").write_bytes(b"x")
    (sub / "note.txt").write_text("x")
    (out / "old_cutout.png").write_bytes(b"x")

    found = scan_inputs([src], recursive=True, exclude_roots=[out])
    assert [p.name for p in found] == ["a.JPG", "b.png"]


def test_scan_inputs_deduplicates_file_and_folder(tmp_path: Path):
    src = tmp_path / "photos"
    src.mkdir()
    image = src / "a.jpg"
    image.write_bytes(b"x")
    found = scan_inputs([src, image], recursive=True)
    assert len(found) == 1


def test_common_source_root_for_multiple_folders(tmp_path: Path):
    a = tmp_path / "one"
    b = tmp_path / "two"
    a.mkdir(); b.mkdir()
    assert common_source_root([a, b]) == tmp_path


def test_suggested_output_dir_for_single_file(tmp_path: Path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    assert suggested_output_dir([image]) == tmp_path / "Background Removed"


def test_suggested_output_dir_for_mixed_sources_uses_common_parent(tmp_path: Path):
    src1 = tmp_path / "one"
    src2 = tmp_path / "two"
    src1.mkdir(); src2.mkdir()
    (src1 / "a.jpg").write_bytes(b"x")
    assert suggested_output_dir([src1, src2 / "b.jpg"]) == tmp_path / "Background Removed"


def test_scan_inputs_accepts_psd_and_psb(tmp_path: Path):
    psd = tmp_path / "layered.psd"
    psb = tmp_path / "large.psb"
    psd.write_bytes(b"x")
    psb.write_bytes(b"x")
    found = scan_inputs([tmp_path], recursive=False)
    assert [p.name for p in found] == ["large.psb", "layered.psd"]


def test_scan_inputs_keeps_explicit_file_inside_excluded_output(tmp_path: Path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    found = scan_inputs([image], recursive=True, exclude_roots=[tmp_path])
    assert found == [image]
