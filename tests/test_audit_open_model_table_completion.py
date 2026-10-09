from tools.audit_open_model_table_completion import ACTIVE_CASES, TABLE7_CASES


def test_completion_audit_scope_matches_requested_models() -> None:
    assert len(TABLE7_CASES) == 12
    assert len(ACTIVE_CASES["matrix_game_2"]) == 14
    assert len(ACTIVE_CASES["matrix_game_3"]) == 12
    assert len(ACTIVE_CASES["lingbot_world_cam"]) == 12
    assert all(not case.startswith("table4/") for case in ACTIVE_CASES["lingbot_world_cam"])
    assert sum(len(cases) for cases in ACTIVE_CASES.values()) == 38
