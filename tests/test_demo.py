from ml import demo


def test_demo_play_classifies_all_eight_tokens():
    labeled = demo._labeled()
    tokens = {tok for (_x, _d, _s, tok) in labeled.values()}
    assert tokens == {"DL", "LB", "CB", "S", "OL", "QB", "RB", "WR"}


def test_demo_render_writes_assets(tmp_path):
    demo.render(tmp_path)
    assert (tmp_path / "demo.png").is_file()
    assert (tmp_path / "demo.mp4").is_file()
    assert (tmp_path / "demo.positions.json").is_file()
