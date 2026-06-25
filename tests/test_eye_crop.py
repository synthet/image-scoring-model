from pathlib import Path

from eye_quality.crop.eye_crop import crop_eye_region, load_oriented_image
from conftest import make_exif_rotated_image, make_synthetic_bird_image


def test_crop_produces_gray_array(tmp_path: Path):
    img = tmp_path / "bird.jpg"
    lx, ly, _, _ = make_synthetic_bird_image(img)
    oriented = load_oriented_image(img)
    crop = crop_eye_region(oriented, lx, ly, head_width_px=60.0, eye_side="left")
    assert crop.gray.size > 0
    assert crop.crop_width_px >= 12


def test_debug_crop_export(tmp_path: Path):
    img = tmp_path / "bird.jpg"
    lx, ly, _, _ = make_synthetic_bird_image(img)
    oriented = load_oriented_image(img)
    debug = tmp_path / "debug"
    crop = crop_eye_region(
        oriented, lx, ly, head_width_px=60.0, debug_dir=debug, eye_side="left"
    )
    assert crop.crop_path is not None
    assert Path(crop.crop_path).is_file()


def test_exif_orientation_applied(tmp_path: Path):
    img = tmp_path / "rotated.jpg"
    expected_x, expected_y = make_exif_rotated_image(img)
    oriented = load_oriented_image(img)
    assert oriented.width == 100
    assert oriented.height == 200
    crop = crop_eye_region(
        oriented, expected_x, expected_y, head_width_px=40.0, eye_side="left"
    )
    assert crop.gray.shape[0] > 0
