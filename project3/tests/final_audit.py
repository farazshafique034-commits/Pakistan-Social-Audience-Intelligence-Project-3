import json
import py_compile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "project3"

REQUIRED = {
    "Project configuration": P3 / "config" / "settings.py",
    "ML configuration": P3 / "config" / "ml_config.py",
    "Dataset generator": P3 / "pipelines" / "generate_dataset.py",
    "Dataset preparation": P3 / "pipelines" / "prepare_dataset.py",
    "Text preprocessing": P3 / "pipelines" / "text_cleaning.py",
    "Feature engineering": P3 / "pipelines" / "feature_engineering.py",
    "ML scaling": P3 / "pipelines" / "ml_preparation.py",
    "Audience clustering": P3 / "pipelines" / "audience_clustering.py",
    "Cluster interpretation": P3 / "analytics" / "cluster_interpretation.py",
    "Cluster evaluation": P3 / "analytics" / "cluster_evaluation.py",
    "Anomaly features": P3 / "pipelines" / "anomaly_features.py",
    "Anomaly detection": P3 / "pipelines" / "anomaly_detection.py",
    "Anomaly evaluation": P3 / "analytics" / "anomaly_evaluation.py",
    "Marketing intelligence": P3 / "analytics" / "marketing_intelligence.py",
    "Dashboard data loader": P3 / "analytics" / "dashboard_data.py",
    "Audience clustering page": ROOT / "pages" / "8_Audience_Clustering.py",
    "Content intelligence page": ROOT / "pages" / "9_Content_Intelligence.py",
    "Anomaly detection page": ROOT / "pages" / "10_Anomaly_Detection.py",
    "Marketing opportunities page": ROOT / "pages" / "11_Marketing_Opportunities.py",
    "Marketing recommendations": P3 / "reports" / "marketing_recommendations.csv",
    "Clustering evaluation": P3 / "reports" / "clustering_evaluation.json",
    "Anomaly evaluation": P3 / "reports" / "anomaly_evaluation.json"
}


def main():
    results = []
    for label, path in REQUIRED.items():
        exists = path.exists()
        status = "PASS" if exists else "MISSING"
        if exists and path.suffix == ".py":
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                status = "SYNTAX_ERROR"
                print(str(exc))
        results.append({
            "component": label,
            "path": str(path.relative_to(ROOT)),
            "status": status
        })
        print(f"[{status}] {label}")

    passed = sum(item["status"] == "PASS" for item in results)
    report = {
        "project": "Pakistan Social Audience Intelligence",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "total_checks": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "all_checks_passed": passed == len(results),
        "scope": "File presence and Python syntax only",
        "results": results
    }

    output = P3 / "reports" / "final_system_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print()
    print("PASSED:", passed, "/", len(results))
    print("Audit report:", output)
    if not report["all_checks_passed"]:
        raise SystemExit(1)
    print("FINAL PROJECT 3 FILE AND SYNTAX AUDIT PASSED")


if __name__ == "__main__":
    main()
