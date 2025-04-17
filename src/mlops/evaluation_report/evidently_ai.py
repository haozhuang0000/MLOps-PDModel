import pandas as pd
import numpy as np
from evidently.future.datasets import DataDefinition
from evidently.future.metrics import *
from evidently.future.presets import *
from sklearn.datasets import make_classification
from evidently.future.datasets import BinaryClassification

from evidently.future.report import Report
from evidently.future.report import Context
from evidently.future.datasets import Dataset
from evidently.future.metric_types import Metric
from evidently.future.metric_types import MetricResult
from evidently.future.metric_types import MetricTestResult
from evidently.future.metric_types import SingleValue
from evidently.future.metric_types import SingleValueTest
from evidently.future.metric_types import SingleValueCalculation
from evidently.future.metric_types import MetricId
from evidently.future.metric_types import SingleValueMetric
from evidently.future.metric_types import TResult
from evidently.future.preset_types import PresetResult
from evidently.future.metric_types import BoundTest
from evidently.future.tests import Reference, eq

from evidently.renderers.html_widgets import plotly_figure
from typing import Optional
from typing import List
from plotly.express import line
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.graph_objects as go

class ArMetric(SingleValueMetric):
    y_true: str
    y_pred_probas_0: str
    y_pred_probas_1: str
    y_pred_probas_2: str

    def _default_tests(self, context: Context) -> List[BoundTest]:
        return [eq(0).bind_single(self.get_fingerprint())]

    def _default_tests_with_reference(self, context: Context) -> List[BoundTest]:
        return [eq(Reference(relative=0.1)).bind_single(self.get_fingerprint())]

# implementation
class ArMetricImplementation(SingleValueCalculation[ArMetric]):
    def calculate(self, context: Context, current_data: Dataset, reference_data: Optional[Dataset]) -> SingleValue:
        y_true = current_data.column(self.metric.y_true).data
        y_pred_0 = current_data.column(self.metric.y_pred_probas_0).data
        y_pred_1 = current_data.column(self.metric.y_pred_probas_1).data
        y_pred_2 = current_data.column(self.metric.y_pred_probas_2).data
        y_pred = pd.concat([pd.DataFrame(y_pred_0), pd.DataFrame(y_pred_1), pd.DataFrame(y_pred_2)], axis=1)
        # Binarize the true labels: 1 if class is 1, else 0
        y_binary = (y_true.astype(int) == 1).astype(int)

        y_true_ref = reference_data.column(self.metric.y_true).data
        y_pred_ref_0 = reference_data.column(self.metric.y_pred_probas_0).data
        y_pred_ref_1 = reference_data.column(self.metric.y_pred_probas_1).data
        y_pred_ref_2 = reference_data.column(self.metric.y_pred_probas_2).data
        y_pred_ref = pd.concat([pd.DataFrame(y_pred_ref_0), pd.DataFrame(y_pred_ref_1), pd.DataFrame(y_pred_ref_2)], axis=1)
        y_binary_ref = (y_true_ref.astype(int) == 1).astype(int)

        # Compute AUC
        auc_A = roc_auc_score(y_binary, y_pred['1'])
        auc_B = roc_auc_score(y_binary_ref, y_pred_ref["1"])
        diff = auc_B - auc_A

        # Compute AR
        ar_A = 2 * auc_A - 1
        ar_B = 2 * auc_B - 1

        result = self.result(value=diff)

        # Compute ROC curves
        fpr_A, tpr_A, _ = roc_curve(y_binary, y_pred['1'])
        fpr_B, tpr_B, _ = roc_curve(y_binary_ref, y_pred_ref["1"])

        # --- First Figure: ROC Curve ---
        # Plot ROC curves
        roc_figure = go.Figure()

        roc_figure.add_trace(go.Scatter(
            x=fpr_A,
            y=tpr_A,
            mode='lines',
            name=f'Model A 0 vs 1(AUC={auc_A:.3f})'
        ))

        roc_figure.add_trace(go.Scatter(
            x=fpr_B,
            y=tpr_B,
            mode='lines',
            name=f'Model B 0 vs 1(AUC={auc_B:.3f})'
        ))

        roc_figure.update_layout(
            title="ROC Curve Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(range=[0, 1]),
            xaxis=dict(range=[0, 1]),
            showlegend=True
        )
        # --- Second Figure: Accuracy Ratio Bar Chart ---
        ar_figure = go.Figure()

        ar_figure.add_trace(go.Bar(
            x=["Model A", "Model B"],
            y=[ar_A, ar_B],
            name="Accuracy Ratio",
            text=[f"{ar_A:.3f}", f"{ar_B:.3f}"],
            textposition='auto'
        ))

        ar_figure.update_layout(
            title="Accuracy Ratio",
            xaxis_title="Model",
            yaxis_title="Accuracy Ratio",
            yaxis=dict(range=[0, 1])
        )

        # --- Assign both figures to result.widget ---
        result.widget = [
            plotly_figure(title="ROC Curve Comparison - 0 vs 1", figure=roc_figure),
            plotly_figure(title="Accuracy Ratio Comparison - 0 vs 1", figure=ar_figure)
        ]
        return result

    def display_name(self) -> str:
        return f"AUC Value"