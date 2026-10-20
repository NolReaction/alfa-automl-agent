"""Exercise frontend decisions without a browser or npm runtime dependencies."""

from pathlib import Path
import shutil
import subprocess
import unittest


NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[2]
CHECKS = Path(__file__).with_name("frontend_checks.cjs")


@unittest.skipUnless(NODE, "Frontend decision checks require Node.js")
class FrontendTests(unittest.TestCase):
    def check_case(self, name):
        result = subprocess.run(
            [NODE, str(CHECKS), name, str(ROOT / "app" / "static" / "app.js")],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_comparison_groups_never_choose_a_winner_across_validation_schemes(self):
        self.check_case("groups")

    def test_comparison_groups_require_current_local_task_data_and_metric(self):
        self.check_case("provenance")

    def test_current_csv_source_is_independent_of_the_best_local_score(self):
        self.check_case("selected_csv")

    def test_terminal_baseline_stays_in_a_collapsed_secondary_report(self):
        self.check_case("terminal_report")

    def test_model_settings_distinguish_defaults_variants_and_missing_evidence(self):
        self.check_case("model_evidence")

    def test_modified_baseline_is_not_used_as_the_standard_reference(self):
        self.check_case("baseline_reference")

    def test_parameter_input_preserves_integer_float_null_and_enum_types(self):
        self.check_case("parameter_types")

    def test_invalid_parameters_are_rejected_before_building_a_request(self):
        self.check_case("parameter_errors")

    def test_edit_and_comparison_inherit_only_the_matching_validation(self):
        self.check_case("validation_inheritance")

    def test_qwen_and_prepare_requests_omit_offline_model_settings(self):
        self.check_case("payload")

    def test_reported_model_settings_are_escaped_and_never_invented(self):
        self.check_case("escaping")


if __name__ == "__main__":
    unittest.main()
