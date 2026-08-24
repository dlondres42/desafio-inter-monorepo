from app import main


def test_main_prints_greeting(capsys):
    main()

    assert capsys.readouterr().out.strip() == "Hello from inference-server!"
