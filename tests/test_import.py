def test_import():
    import picomap
    assert hasattr(picomap, "build_map")
