import pytest
from tools.audit_production_readiness import build_report, render_markdown


@pytest.mark.integration("case_data")
def test_production_readiness_manifest_covers_live_suite() -> None:
    report = build_report()

    assert report["status"] == "valid"
    assert report["manifest_errors"] == []
    assert report["summary"]["dimension_count"] == 8
    assert report["summary"]["case_count"] == 34
    assert report["summary"]["native_trajectory_case_count"] == 32
    assert report["summary"]["supplemental_trajectory_case_count"] == 2
    assert report["summary"]["open_priority_counts"] == {
        "P0": 4,
        "P1": 7,
        "P2": 5,
    }
    assert {row["dimension"] for row in report["dimensions"]} == {
        "lpips",
        "action_controllability",
        "prompt_scene_consistency",
        "lighting_shading",
        "environment_object_consistency",
        "kinematics_mechanics",
        "physical_causality",
        "fluid_mechanics",
    }


@pytest.mark.integration("case_data")
def test_production_readiness_report_surfaces_portability_gaps() -> None:
    report = build_report()
    rows = {row["dimension"]: row for row in report["dimensions"]}

    # The 12 Table 7 cells are native Lite cases; the two remaining
    # supplemental rows are the legacy ChessBoard walk and indoor Garden cases.
    assert rows["environment_object_consistency"]["trajectory_coverage"] == {
        "native": 16,
        "supplemental": 0,
        "other": 0,
    }
    assert rows["kinematics_mechanics"]["trajectory_coverage"] == {
        "native": 3,
        "supplemental": 0,
        "other": 0,
    }
    assert rows["fluid_mechanics"]["project_genie_executable_cases"] == 4
    assert rows['lighting_shading']['readiness'] == 'ready'
    assert rows['lighting_shading']['evaluator_lock_nonempty_case_counts']['dependency_versions'] == 1
    # Every live case now carries a dependency lock: lighting_orbit_v2's
    # hand-pinned GPU environment, every other case written by
    # tools/lock_case_dependencies.py.
    assert all(
        row["evaluator_lock_nonempty_case_counts"]["dependency_versions"] == row["case_count"]
        for row in report["dimensions"]
    )
    markdown = render_markdown(report)
    assert "A valid case bundle is not automatically" in markdown
    assert "First-person RacingCar ego translation" not in markdown
    assert "Untracked labels can disappear" not in markdown
    assert "aggregate can mix causal LPIPS" not in markdown
    assert "`dependency_lock_content`" in markdown
    assert "P0 / P1 / P2" in markdown
