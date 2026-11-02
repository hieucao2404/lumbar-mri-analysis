"""Guard against silently dropping failures or mapping protrusion to normal."""

from scripts.evaluate_lsma_paired_pipeline import LABEL_MAP, summarize


def row(truth, prediction):
    return {
        "truth": truth, "source_label": "Diffuse Bulge" if truth else "None",
        "final_label": prediction, "status": prediction,
        "decision_reason": prediction,
    }


def test_indeterminate_cases_remain_in_coverage_and_full_cohort_sensitivity():
    result = summarize([
        row(1, "abnormal"), row(0, "abnormal"), row(0, "normal"),
        row(1, "indeterminate"), row(0, "indeterminate"),
    ])
    assert result["labelled_discs"] == 5
    assert result["decided_discs"] == 3
    assert result["decision_coverage"] == 3 / 5
    assert result["indeterminate_abnormal"] == 1
    assert result["indeterminate_normal"] == 1
    assert result["final_on_decided_discs"]["sensitivity"] == 1
    assert result["final_on_decided_discs"]["f1"] == 2 / 3
    assert result["detected_abnormal_sensitivity_all_labelled_discs"] == 1 / 2


def test_no_decisions_does_not_produce_a_perfect_score():
    result = summarize([row(1, "indeterminate"), row(0, "indeterminate")])
    assert result["decision_coverage"] == 0
    assert result["final_on_decided_discs"]["f1"] is None
    assert result["detected_abnormal_sensitivity_all_labelled_discs"] == 0


def test_protrusion_is_abnormal_and_unknown_labels_are_not_normal():
    assert LABEL_MAP["Central Protrusion"] == 1
    assert LABEL_MAP.get("unknown") is None


def test_axial_effect_uses_the_same_decided_discs_for_both_stages():
    rows = [row(0, "normal"), row(1, "normal"), row(1, "abnormal"),
            row(0, "indeterminate")]
    for item in rows:
        item["sagittal_flag"] = True
    result = summarize(rows)
    assert result["sagittal_on_decided_discs"]["discs"] == 3
    assert result["sagittal_on_decided_discs"]["tp"] == 2
    assert result["sagittal_on_decided_discs"]["fp"] == 1
    assert result["axial_effect_on_decided_discs"] == {
        "false_positive_alerts_corrected": 1,
        "true_positive_alerts_lost": 1,
    }
