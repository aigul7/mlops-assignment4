import pandas as pd
import numpy as np
import os
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset

scenario = os.environ.get('SCENARIO', 'baseline')
print(f"Running monitoring for scenario: {scenario}")

reference = pd.read_csv('data/predictions_datarobot_baseline.csv', encoding='latin-1')
pred_col = [c for c in reference.columns if 'PREDICTION' in c.upper()]
reference['prediction'] = reference[pred_col[0]].values
print(f"Reference loaded: {len(reference)} rows")

if scenario == 'baseline':
    current = reference.copy()
else:
    scenario_map = {
        'A': 'data/modified/predictions_datarobot_A.csv',
        'AB': 'data/modified/predictions_datarobot_AB.csv',
        'ABC': 'data/modified/predictions_datarobot_ABC.csv',
    }
    current = pd.read_csv(scenario_map[scenario])
    current['prediction'] = current['TARGET_deathRate_PREDICTION'].values
    current['TARGET_deathRate'] = reference['TARGET_deathRate'].values
    print(f"Current loaded: {len(current)} rows")

column_mapping = ColumnMapping(
    target='TARGET_deathRate',
    prediction='prediction'
)

report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(), DataQualityPreset()])
report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

os.makedirs('reports', exist_ok=True)
output_path = f'reports/docker_monitoring_{scenario}.html'
report.save_html(output_path)
print(f"Report saved: {output_path}")
