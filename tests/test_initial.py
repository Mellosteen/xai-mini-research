from xai_mini_research import get_default_config_path

def test_default_config():
    config_path = get_default_config_path()

    assert config_path.exists()
    assert config_path.name == "default.yaml"