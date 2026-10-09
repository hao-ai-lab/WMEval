from tools import finalize_open_model_tables as finalizer


def test_readiness_checks_exact_active_scope(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(
        finalizer,
        "rows_for",
        lambda model, case_id: seen.append((model, case_id)) or [{}] * 5,
    )
    monkeypatch.setattr(finalizer, "table4_block", lambda _model: ("block", {}))
    monkeypatch.setattr(finalizer, "table7_block", lambda _model: ("block", {}))

    assert finalizer.readiness_errors() == []
    assert len(seen) == 38
    assert not any(
        model == "lingbot_world_cam" and case.startswith("table4/")
        for model, case in seen
    )


def test_readiness_reports_incomplete_cases(monkeypatch) -> None:
    def incomplete(model: str, case_id: str):
        if model == "lingbot_world_cam":
            raise ValueError("waiting for gate")
        return [{}] * 5

    monkeypatch.setattr(finalizer, "rows_for", incomplete)
    monkeypatch.setattr(finalizer, "table4_block", lambda _model: ("block", {}))
    monkeypatch.setattr(finalizer, "table7_block", lambda _model: ("block", {}))

    errors = finalizer.readiness_errors()
    assert len(errors) == 12
    assert all("lingbot_world_cam" in error for error in errors)
