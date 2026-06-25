import json
from pathlib import Path

from training.prepare_dataset import _coco_to_yolo_line


def test_coco_to_yolo_line_format():
    line = _coco_to_yolo_line(
        bbox_xywh=[100, 50, 80, 60],
        keypoints=[120, 70, 2, 160, 70, 2] + [0, 0, 0] * 4,
        img_w=400,
        img_h=300,
    )
    parts = line.split()
    assert parts[0] == "0"
    assert len(parts) == 5 + 6 * 3


def test_prepare_csv_roundtrip(tmp_path: Path):
    from PIL import Image

    from training.prepare_dataset import prepare_from_csv

    img = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 80), color=(100, 120, 140)).save(img)
    csv_path = tmp_path / "ann.csv"
    csv_path.write_text(
        "image_path,width,height,bbox_json,keypoints_json\n"
        f"{img},100,80,"
        '"[20,10,60,50]",'
        '"[40,30,2,60,30,2,0,0,0,0,0,0,0,0,0,0,0,0]"\n',
        encoding="utf-8",
    )
    out = tmp_path / "dataset"
    prepare_from_csv(csv_path, out, val_ratio=0.0)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    label = Path(manifest[0]["label"])
    assert label.is_file()
