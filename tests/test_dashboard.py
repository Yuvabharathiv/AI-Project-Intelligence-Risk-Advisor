import unittest

from project_agents import (
    analyze_project_document,
    calculate_project_health,
    generate_project_documentation,
)
from ui import (
    _build_report_markdown,
    _dashboard_recommendations,
    _risk_counts,
)


class DashboardDataTests(unittest.TestCase):
    def setUp(self):
        self.analysis = analyze_project_document(
            "Project goal: Deliver the analytics portal.\n"
            "Status: API dependency is delayed.\n"
            "Blocker: Deployment is blocked.\n"
            "Action item: Owner: Alice, follow up with the API team.",
            "plan.txt",
        )

    def test_risk_summary_only_includes_supported_severity_levels(self):
        risks = self.analysis["delivery_risk"]["risks"]
        counts = _risk_counts(risks)

        self.assertEqual(counts, {"high": 2, "medium": 0})

    def test_recommendations_retain_sources_and_documented_actions(self):
        totals = {
            "risks": self.analysis["delivery_risk"]["risks"],
            "blockers": self.analysis["blockers_and_actions"]["blockers"],
            "decisions": [],
            "actions": self.analysis["blockers_and_actions"]["actions"],
        }

        recommendations = _dashboard_recommendations(
            [self.analysis], totals
        )

        risk_recommendation = next(
            item for item in recommendations if item["priority"] == "High"
        )
        self.assertEqual(risk_recommendation["source"], "plan.txt")
        self.assertIn("follow up with the API team", risk_recommendation["detail"])
        self.assertTrue(
            any("blocked" in item["text"] for item in recommendations)
        )
        self.assertTrue(
            any("missing deliverable" in item["text"] for item in recommendations)
        )

    def test_export_contains_health_risks_blockers_actions_and_recommendations(self):
        analyses = [self.analysis]
        documentation = generate_project_documentation(analyses)
        health = calculate_project_health(analyses)

        report = _build_report_markdown(
            documentation,
            health,
            analyses,
            project_name="Analytics portal",
        )

        self.assertIn("# Analytics portal", report)
        self.assertIn("## Health scoring", report)
        self.assertIn("## Risk register", report)
        self.assertIn("## Blockers", report)
        self.assertIn("## Action items", report)
        self.assertIn("## Project recommendations", report)
        self.assertIn("plan.txt", report)


if __name__ == "__main__":
    unittest.main()
