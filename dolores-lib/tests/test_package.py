from importlib.metadata import version

from dolores import main


def test_distribution_is_installed():
    assert version("dolores-lib")


def test_main_prints_greeting(capsys):
    main()

    assert capsys.readouterr().out.strip() == "Hello from dolores-lib!"
