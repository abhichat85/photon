# tests/test_dataprep.py
import json

from photon.finetune.dataprep import prepare, scrub_pii, validate_example

GOOD = {
    "messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
}


def test_scrub_replaces_email():
    assert scrub_pii("mail me at jane.doe+x@acme.co please") == "mail me at [EMAIL] please"


def test_scrub_replaces_phone():
    assert scrub_pii("call +1 415-555-0134 now") == "call [PHONE] now"


def test_validate_rejects_bad_examples():
    assert validate_example(GOOD)
    assert not validate_example({"messages": [{"role": "user", "content": "no reply"}]})
    assert not validate_example({"messages": [{"role": "wizard", "content": "x"},
                                              {"role": "assistant", "content": "y"}]})
    assert not validate_example({"nope": True})


def test_prepare_dedupes_and_reports(tmp_path):
    raw = tmp_path / "raw.jsonl"
    lines = [json.dumps(GOOD)] * 3 + ["not json"] + [
        json.dumps({"messages": [{"role": "user", "content": "q2"},
                                 {"role": "assistant", "content": "a2"}]})
    ]
    raw.write_text("\n".join(lines))
    report = prepare(raw, tmp_path / "out", eval_fraction=0.0, seed=7)
    assert report.total == 5
    assert report.dropped_invalid == 1
    assert report.dropped_duplicate == 2
    assert report.kept == 2
    assert report.train == 2 and report.eval == 0


def test_prepare_split_is_deterministic(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps({"messages": [{"role": "user", "content": f"q{i}"},
                                     {"role": "assistant", "content": f"a{i}"}]})
            for i in range(20)
        )
    )
    prepare(raw, tmp_path / "out1", eval_fraction=0.2, seed=7)
    prepare(raw, tmp_path / "out2", eval_fraction=0.2, seed=7)
    assert (tmp_path / "out1" / "train.jsonl").read_text() == (
        tmp_path / "out2" / "train.jsonl"
    ).read_text()
    assert (tmp_path / "out1" / "eval.jsonl").read_text() == (
        tmp_path / "out2" / "eval.jsonl"
    ).read_text()
