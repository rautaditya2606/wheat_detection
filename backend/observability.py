import os
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from models import Feedback

def get_drift_report_html():
    """Generates an Evidently Data & Target Drift report comparing reference to production predictions."""
    reference_path = os.path.join(os.path.dirname(__file__), "static", "reference_data.csv")
    if not os.path.exists(reference_path):
        return "<h3>Reference dataset not found. Please pre-generate the reference CSV.</h3>"

    # 1. Load reference dataset (filter to columns present in current_data to avoid partial column errors)
    ref_df = pd.read_csv(reference_path)[["prediction", "confidence"]]

    # 2. Load current production predictions
    all_feedback = Feedback.query.all()
    if not all_feedback:
        return "<h3>No production predictions recorded yet. Submit feedback to see drift analysis.</h3>"

    current_records = []
    for f in all_feedback:
        current_records.append({
            "prediction": f.predicted_class,
            "confidence": f.confidence,
        })
    curr_df = pd.DataFrame(current_records)

    # Define ColumnMapping for prediction and confidence
    column_mapping = ColumnMapping()
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = ["confidence"]

    report = Report(metrics=[DataDriftPreset(columns=["prediction", "confidence"])])
    report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)
    return report.get_html()

def get_performance_report_html():
    """Generates an Evidently Classification Performance report based on admin-verified feedback."""
    reference_path = os.path.join(os.path.dirname(__file__), "static", "reference_data.csv")
    if not os.path.exists(reference_path):
        return "<h3>Reference dataset not found. Please pre-generate the reference CSV.</h3>"

    # 1. Load reference dataset
    ref_df = pd.read_csv(reference_path)

    # 2. Load verified production feedback
    verified_feedback = Feedback.query.filter_by(is_verified=True).all()
    if len(verified_feedback) < 3:
        return (
            "<h3>Insufficient verified data.</h3>"
            f"<p>Currently, there are only {len(verified_feedback)} verified feedback records. "
            "At least 3 verified records are required to generate the classification performance report.</p>"
            "<p>To add verified feedback, go to the Admin panel and verify uploaded predictions.</p>"
        )

    current_records = []
    for f in verified_feedback:
        # Target is the correct class if incorrect and correct_class is specified, else predicted_class
        target_class = f.correct_class if (f.correct_class and not f.is_correct) else f.predicted_class
        current_records.append({
            "target": target_class,
            "prediction": f.predicted_class,
            "confidence": f.confidence,
        })
    curr_df = pd.DataFrame(current_records)

    # Define ColumnMapping mapping
    column_mapping = ColumnMapping()
    column_mapping.target = "target"
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = ["confidence"]

    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=ref_df, current_data=curr_df, column_mapping=column_mapping)
    return report.get_html()
