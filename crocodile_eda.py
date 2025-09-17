"""Comprehensive EDA for crocodile dataset.

Generates:
 - Console summaries
 - Saved summary CSVs in eda_outputs/
 - Visualization PNGs in eda_outputs/figures/
 - Optional profiling report (if ydata_profiling installed)
 - Simple predictive model (length -> weight)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from scipy import stats
import json
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent / 'crocodile_dataset.csv'
OUT_DIR = Path(__file__).parent / 'eda_outputs'
FIG_DIR = OUT_DIR / 'figures'

OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Dataset not found at {path}")
	df = pd.read_csv(path)
	if 'Date of Observation' in df.columns:
		df['Date of Observation'] = pd.to_datetime(df['Date of Observation'], errors='coerce', dayfirst=True)
		df['Year'] = df['Date of Observation'].dt.year
		df['Month'] = df['Date of Observation'].dt.month
	return df


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
	num_cols = [c for c in df.columns if 'Observed' in c and df[c].dtype != 'O']
	summary = df[num_cols].describe().T
	summary['median'] = df[num_cols].median()
	summary['variance'] = df[num_cols].var()
	summary['skew'] = df[num_cols].skew()
	summary['kurtosis'] = df[num_cols].kurtosis()
	return summary


def categorical_summary(df: pd.DataFrame) -> Dict[str, pd.Series]:
	cat_cols = ['Common Name','Scientific Name','Family','Genus','Age Class','Sex','Country/Region','Habitat Type','Conservation Status']
	result = {}
	for col in cat_cols:
		if col in df.columns:
			result[col] = df[col].value_counts()
	return result


def data_quality(df: pd.DataFrame) -> Dict[str, object]:
	missing = df.isnull().sum()
	duplicates = df.duplicated().sum()
	dup_ids = df['Observation ID'].duplicated().sum() if 'Observation ID' in df.columns else None
	return {
		'missing_non_zero': missing[missing > 0].to_dict(),
		'total_missing': int(missing.sum()),
		'duplicate_rows': int(duplicates),
		'duplicate_ids': int(dup_ids) if dup_ids is not None else None
	}


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
	metrics = []
	num_cols = [c for c in ['Observed Length (m)','Observed Weight (kg)'] if c in df.columns]
	for col in num_cols:
		q1 = df[col].quantile(0.25)
		q3 = df[col].quantile(0.75)
		iqr = q3 - q1
		lower = q1 - 1.5 * iqr
		upper = q3 + 1.5 * iqr
		mask = (df[col] < lower) | (df[col] > upper)
		metrics.append({'feature': col,'q1': q1,'q3': q3,'iqr': iqr,'lower': lower,'upper': upper,'outlier_count': int(mask.sum())})
	return pd.DataFrame(metrics)


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
	cols = [c for c in ['Observed Length (m)','Observed Weight (kg)','Year'] if c in df.columns]
	return df[cols].corr() if cols else pd.DataFrame()


def grouped_aggregations(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
	results = {}
	if 'Common Name' in df.columns:
		results['top_species_by_length'] = (df.groupby('Common Name')['Observed Length (m)']
											  .mean().sort_values(ascending=False).head(10).to_frame('avg_length_m'))
	if 'Conservation Status' in df.columns:
		results['conservation_status_counts'] = df['Conservation Status'].value_counts().to_frame('count')
	if 'Country/Region' in df.columns:
		results['top_countries'] = df['Country/Region'].value_counts().head(10).to_frame('count')
	return results


def length_weight_regression(df: pd.DataFrame) -> Dict[str, float]:
	if not {'Observed Length (m)','Observed Weight (kg)'} <= set(df.columns):
		return {}
	x = df['Observed Length (m)']
	y = df['Observed Weight (kg)']
	slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
	return {
		'slope': slope,
		'intercept': intercept,
		'r_squared': r_value**2,
		'p_value': p_value,
		'std_err': std_err,
		'pearson_corr': float(x.corr(y))
	}


def save_dataframe(df: pd.DataFrame, name: str):
	path = OUT_DIR / f'{name}.csv'
	df.to_csv(path, index=True)
	print(f'Saved: {path}')


def save_series_dict(d: Dict[str, pd.Series], prefix: str):
	for k, s in d.items():
		path = OUT_DIR / f'{prefix}_{k.replace(" ", "_").lower()}.csv'
		s.to_csv(path)
		print(f'Saved: {path}')


def build_visuals(df: pd.DataFrame):
	sns.set_theme(style='whitegrid')
	# Histograms
	for col in ['Observed Length (m)','Observed Weight (kg)']:
		if col in df.columns:
			plt.figure(figsize=(8,4))
			sns.histplot(df[col], kde=True, bins=30, color='#2a9d8f')
			plt.title(f'Distribution of {col}')
			plt.tight_layout()
			plt.savefig(FIG_DIR / f'hist_{col.replace(" ", "_").replace("(", "").replace(")", "")}.png')
			plt.close()
	# Boxplots by Conservation Status
	if {'Conservation Status','Observed Length (m)'} <= set(df.columns):
		plt.figure(figsize=(10,5))
		sns.boxplot(data=df, x='Conservation Status', y='Observed Length (m)', palette='Set3')
		plt.xticks(rotation=30, ha='right')
		plt.title('Length by Conservation Status')
		plt.tight_layout()
		plt.savefig(FIG_DIR / 'box_length_by_status.png')
		plt.close()
	# Scatter Length vs Weight
	if {'Observed Length (m)','Observed Weight (kg)'} <= set(df.columns):
		plt.figure(figsize=(6,6))
		sns.scatterplot(data=df, x='Observed Length (m)', y='Observed Weight (kg)', hue='Common Name', legend=False, alpha=0.6)
		plt.title('Length vs Weight')
		plt.tight_layout()
		plt.savefig(FIG_DIR / 'scatter_length_weight.png')
		plt.close()
	# Correlation Heatmap
	corr = correlation_analysis(df)
	if not corr.empty:
		plt.figure(figsize=(5,4))
		sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar_kws={'shrink':0.8})
		plt.title('Correlation Matrix')
		plt.tight_layout()
		plt.savefig(FIG_DIR / 'correlation_matrix.png')
		plt.close()


def optional_profile(df: pd.DataFrame):
	try:
		from ydata_profiling import ProfileReport
	except ImportError:
		print('ydata_profiling not installed; skipping profile. Install with: pip install ydata-profiling')
		return
	profile = ProfileReport(df, title='Crocodile Dataset Profile', minimal=True)
	out_path = OUT_DIR / 'profile_report.html'
	profile.to_file(out_path)
	print(f'Saved profiling report: {out_path}')


def main():
	print('=== LOADING DATA ===')
	df = load_data(DATA_PATH)
	print(f'Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')

	print('\n=== NUMERIC SUMMARY ===')
	num_sum = numeric_summary(df)
	print(num_sum)
	save_dataframe(num_sum, 'numeric_summary')

	print('\n=== CATEGORICAL SUMMARY ===')
	cat_sum = categorical_summary(df)
	for k,v in cat_sum.items():
		print(f'\n{k}\n{v.head(10)}')
	save_series_dict(cat_sum, 'freq')

	print('\n=== DATA QUALITY ===')
	dq = data_quality(df)
	print(dq)
	with open(OUT_DIR / 'data_quality.json','w') as f:
		json.dump(dq, f, indent=2)
	print(f'Saved: {OUT_DIR / "data_quality.json"}')

	print('\n=== OUTLIERS ===')
	outliers = detect_outliers_iqr(df)
	print(outliers)
	save_dataframe(outliers, 'outliers_iqr')

	print('\n=== CORRELATION ===')
	corr = correlation_analysis(df)
	if not corr.empty:
		print(corr)
		save_dataframe(corr, 'correlation_matrix')

	print('\n=== GROUPED AGGREGATIONS ===')
	groups = grouped_aggregations(df)
	for name, gdf in groups.items():
		print(f'\n{name}\n{gdf.head(10)}')
		save_dataframe(gdf, name)

	print('\n=== LENGTH-WEIGHT REGRESSION ===')
	reg = length_weight_regression(df)
	print(reg)
	with open(OUT_DIR / 'length_weight_regression.json','w') as f:
		json.dump(reg, f, indent=2)
	print(f'Saved: {OUT_DIR / "length_weight_regression.json"}')

	print('\n=== BUILDING VISUALIZATIONS ===')
	build_visuals(df)
	print(f'Figures saved to {FIG_DIR}')

	print('\n=== OPTIONAL PROFILING (if installed) ===')
	optional_profile(df)

	print('\n=== EDA COMPLETE ===')


if __name__ == '__main__':
	main()
	
import plotly.express as px
import pandas as pd

