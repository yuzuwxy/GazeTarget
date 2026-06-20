from PIL import Image

from preprocessing.image_source import discover_images


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(path)


def test_directory_discovery_is_recursive_and_stable(tmp_path):
    _write_image(tmp_path / "b.jpg")
    _write_image(tmp_path / "nested" / "a.png")
    (tmp_path / "ignored.txt").write_text("x", encoding="utf-8")

    records = discover_images(
        {"type": "directory", "path": str(tmp_path), "extensions": [".jpg", ".png"]}
    )

    assert [record.relative_path for record in records] == ["b.jpg", "nested/a.png"]
    assert len({record.image_id for record in records}) == 2


def test_gazefollow_discovery_deduplicates_annotation_rows(tmp_path):
    data_extended = tmp_path / "data_extended"
    _write_image(data_extended / "train" / "a.jpg")
    _write_image(data_extended / "train" / "b.jpg")
    annotation = data_extended / "train_annotations_release.txt"
    annotation.write_text(
        "train/a.jpg,1,0,0,1,1,0,0,0,0,0,0,1,1,1,x,y\n"
        "train/a.jpg,2,0,0,1,1,0,0,0,0,0,0,1,1,1,x,y\n"
        "train/b.jpg,3,0,0,1,1,0,0,0,0,0,0,1,1,1,x,y\n",
        encoding="utf-8",
    )

    records = discover_images(
        {"type": "gazefollow", "path": str(tmp_path), "split": "train"}
    )

    assert [record.relative_path for record in records] == ["train/a.jpg", "train/b.jpg"]
