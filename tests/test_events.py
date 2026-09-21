from sfx.events import EventStream


def test_emit_fans_to_all_sinks():
    a, b = [], []
    s = EventStream()
    s.subscribe(a.append)
    s.subscribe(b.append)

    s.emit({"ev": "predict", "pred": "edit"})

    assert a == [{"ev": "predict", "pred": "edit"}]
    assert b == [{"ev": "predict", "pred": "edit"}]


def test_stream_is_append_only():
    s = EventStream()
    s.emit({"ev": "spec_launch", "id": "c1"})
    s.emit({"ev": "serve", "id": "c1"})

    assert s.events == [
        {"ev": "spec_launch", "id": "c1"},
        {"ev": "serve", "id": "c1"},
    ]
