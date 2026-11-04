"""RAW embedded-preview extraction on synthetic TIFF/NEF layouts (no camera files needed)."""
from __future__ import annotations

import io
import struct

from PIL import Image

from training.build_eye_dataset import _extract_by_marker_scan, _extract_from_subifd, extract_full_preview


def _jpeg(size=(64, 48), noisy=False) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    if noisy:  # realistic size: the marker scan ignores JPEGs under 10 kB
        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    else:
        img = Image.new("RGB", size, (120, 80, 40))
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _entry(tag, typ, count, value) -> bytes:
    return struct.pack(">HHII", tag, typ, count, value)


def _nef(subifd_jpegs: list[bytes], inline_single: bool) -> bytes:
    """Big-endian TIFF: IFD0 with a SubIFDs tag; each SubIFD points at one embedded JPEG."""
    header = b"MM" + struct.pack(">HI", 42, 8)
    n_sub = len(subifd_jpegs)
    ifd0_size = 2 + 12 + 4
    array_off = 8 + ifd0_size
    sub_base = array_off + (0 if inline_single and n_sub == 1 else 4 * n_sub)
    sub_size = 2 + 2 * 12 + 4
    data_off = sub_base + sub_size * n_sub
    sub_offsets, jpeg_offsets, off = [], [], data_off
    for i, j in enumerate(subifd_jpegs):
        sub_offsets.append(sub_base + i * sub_size)
        jpeg_offsets.append(off)
        off += len(j)
    value = sub_offsets[0] if inline_single and n_sub == 1 else array_off
    out = header + struct.pack(">H", 1) + _entry(0x014A, 4, n_sub, value) + struct.pack(">I", 0)
    if not (inline_single and n_sub == 1):
        out += struct.pack(f">{n_sub}I", *sub_offsets)
    for j, jo in zip(subifd_jpegs, jpeg_offsets):
        out += struct.pack(">H", 2) + _entry(0x0201, 4, 1, jo) + _entry(0x0202, 4, 1, len(j)) + struct.pack(">I", 0)
    for j in subifd_jpegs:
        out += j
    return out


def test_single_subifd_inline_offset():
    small = _jpeg((32, 24))
    assert _extract_from_subifd(_nef([small], inline_single=True)) == small


def test_multiple_subifds_read_through_the_offset_array():
    # Nikon D90 layout: SubIFDs count 2, value points to an array of offsets; the larger JPEG wins
    small, large = _jpeg((32, 24)), _jpeg((640, 480))
    assert _extract_from_subifd(_nef([small, large], inline_single=False)) == large


def test_marker_scan_skips_false_soi_without_a_following_marker():
    real = _jpeg((320, 240), noisy=True)
    assert len(real) > 10000
    junk = b"\x00" * 600 + b"\xff\xd8\x9f\xce" + b"\x11" * 20000 + b"\xff\xd9"  # FF D8 inside raw data
    found = _extract_by_marker_scan(junk + real)
    assert found == real
    Image.open(io.BytesIO(found)).load()


def test_extract_full_preview_returns_a_decodable_jpeg(tmp_path):
    small, large = _jpeg((32, 24)), _jpeg((640, 480))
    p = tmp_path / "x.NEF"
    p.write_bytes(_nef([small, large], inline_single=False))
    jpg, method = extract_full_preview(p)
    assert method == "subifd" and Image.open(io.BytesIO(jpg)).size == (640, 480)
