from origami_texturing import paths


def test_project_root_contains_pyproject():
    assert (paths.ROOT_DIR / "pyproject.toml").is_file()


def test_data_paths_are_inside_project():
    assert paths.DATA_DIR.parent == paths.ROOT_DIR
    assert paths.RAW_DATA_DIR.parent == paths.DATA_DIR
    assert paths.PROCESSED_DATA_DIR.parent == paths.DATA_DIR


def test_external_path_is_inside_project():
    assert paths.EXTERNAL_DIR.parent == paths.ROOT_DIR
