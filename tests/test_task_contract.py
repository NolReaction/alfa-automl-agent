import unittest

from agent.task_contract import validate_task_metric


class TaskMetricContractTests(unittest.TestCase):
    def test_supported_english_and_russian_declarations(self):
        for declaration, classification in (
            ("Primary metric: Macro F1", True),
            ("Main metric is macro-averaged F1 score (higher is better).", True),
            ("Основная метрика — **Macro F1**:", True),
            ("Metric: RMSE", False),
            ("Evaluation metric = root mean squared error (RMSE).", False),
            ("Метрика оценки — **RMSE** на исходной шкале таргета:", False),
            ("## Метрика оценки\n\nRMSE (меньше — лучше)", False),
            ("## Primary metric\n\n**Macro F1**", True),
        ):
            with self.subTest(declaration=declaration):
                self.assertEqual(validate_task_metric(declaration, classification), {
                    "name": "Macro F1" if classification else "RMSE",
                    "higher_is_better": classification, "source": "task.md",
                })

    def test_supplementary_metrics_do_not_override_primary(self):
        text = (
            "Primary metric: Macro F1. Accuracy is only supplementary.\n"
            "Additional metrics: Accuracy and Weighted F1.\n"
            "Дополнительно рассчитываются Accuracy и Weighted F1.\n"
        )
        self.assertEqual(validate_task_metric(text, True)["name"], "Macro F1")

    def test_unsupported_primary_is_not_rescued_by_supported_secondary(self):
        for text, classification in (
            ("Primary metric: Accuracy\nAdditional metrics: Macro F1", True),
            ("Основная метрика — Weighted F1\nДополнительно: Macro F1", True),
            ("Metric: MAE\nCompare RMSE for diagnostics.", False),
            ("Метрика оценки: RMSLE\nRMSE is not the evaluation metric.", False),
            ("Primary metric: F1", True),
            ("Primary metric: ROC AUC", True),
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    validate_task_metric(text, classification)

    def test_missing_or_ambiguous_metric_never_falls_back_to_target_type(self):
        for text in (
            "Predict the numeric target. Models often use RMSE.",
            "Metric: RMSE or MAE", "Metric: RMSE/MAE", "Metric: RMSE, MAE",
            "Metric: RMSE; MAE",
            "Metric: RMSE\nPrimary metric: Macro F1",
            "Primary metric:", "Metric: RMSE_custom",
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    validate_task_metric(text, False)

    def test_declared_metric_must_match_target_type(self):
        for text, classification in (("Metric: RMSE", True), ("Metric: Macro F1", False)):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "but this offline"):
                    validate_task_metric(text, classification)

    def test_examples_and_comments_are_not_primary_declarations(self):
        text = (
            "```text\nPrimary metric: Accuracy\n```\n"
            "<!-- Metric: MAE -->\n"
            "    Metric: RMSLE\n"
            "Primary metric: Macro F1\n"
        )
        self.assertEqual(validate_task_metric(text, True)["name"], "Macro F1")
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_task_metric("```text\nMetric: RMSE\n```", False)

    def test_public_task_excerpts(self):
        # Keep meaningful public-task coverage in CI without downloading data.
        classification = (
            "Классы существенно несбалансированы. Поэтому основной\n"
            "метрикой выбран Macro F1, а не Accuracy.\n\n"
            "Основная метрика — **Macro F1**:\n\n"
            "```python\nf1_score(y_true, y_pred, average=\"macro\")\n```\n\n"
            "Дополнительно рассчитываются **Accuracy** и **Weighted F1**.\n"
        )
        regression = (
            "Метрика оценки — **RMSE** на исходной шкале таргета:\n"
            "`sqrt(mean((y_true - y_pred) ** 2))`. Меньшее значение лучше.\n"
        )
        self.assertEqual(validate_task_metric(classification, True)["name"], "Macro F1")
        self.assertEqual(validate_task_metric(regression, False)["name"], "RMSE")


if __name__ == "__main__":
    unittest.main()
