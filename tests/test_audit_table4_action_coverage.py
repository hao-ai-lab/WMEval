from __future__ import annotations

import tools.audit_table4_action_coverage as module


def test_table4_action_audit_covers_all_views_and_repeats() -> None:
    report = module.build()

    assert report["status"] == "complete"
    assert report["table4_case_count"] == 2
    assert report["valid_generation_count"] == 10
    assert report["cross_view_action_identity"] == {
        "RacingCar": True,
    }


def test_channel_coverage_and_declared_semantic_gaps() -> None:
    report = module.build()
    racing = report["cases"]["table4/RacingCar/third"]
    billiard = report["cases"]["table4/BilliardGame/first"]

    assert racing["trajectory"]["channels"]["steer"]["missing_states"] == []
    assert racing["trajectory"]["channels"]["throttle"]["missing_states"] == [
        "negative"
    ]
    assert billiard["trajectory"]["channels"]["lateral"]["missing_states"] == []
    assert billiard["trajectory"]["channels"]["forward"]["missing_states"] == [
        "negative"
    ]
    assert billiard["evaluation_status"] == "diagnostic_only_camera_pose_leakage_risk"
    assert "table4/BilliardGame/third" not in report["cases"]
