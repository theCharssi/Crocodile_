import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

DATA_PATH = Path('Crocodile') / 'crocodile_dataset.csv'
EDA_DIR = Path('Crocodile') / 'eda_outputs'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if 'Date of Observation' in df.columns:
        df['Date of Observation'] = pd.to_datetime(df['Date of Observation'], errors='coerce', dayfirst=True)
        df['Year'] = df['Date of Observation'].dt.year
        df['Month'] = df['Date of Observation'].dt.month
    return df

@st.cache_data
def load_optional_csv(path: Path):
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()

@st.cache_data
def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return {}

df = load_data()

st.set_page_config(page_title="Crocodile Dashboard", layout="wide", page_icon="🐊")
st.title("🐊 Crocodile Dataset Interactive Dashboard")
st.markdown("Explore species metrics, conservation status, geography, and temporal patterns. Data sourced from the crocodile dataset.")

# Sidebar Filters
st.sidebar.header("Filters")
species = ['All'] + sorted(df['Common Name'].dropna().unique().tolist())
selected_species = st.sidebar.selectbox('Species', species)

statuses = ['All'] + sorted(df['Conservation Status'].dropna().unique().tolist())
selected_status = st.sidebar.selectbox('Conservation Status', statuses)

countries = ['All'] + sorted(df['Country/Region'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox('Country/Region', countries)

if 'Year' in df.columns:
    year_min, year_max = int(df['Year'].min()), int(df['Year'].max())
    year_range = st.sidebar.slider('Year Range', year_min, year_max, (year_min, year_max))
else:
    year_range = None

filtered = df.copy()
if selected_species != 'All':
    filtered = filtered[filtered['Common Name'] == selected_species]
if selected_status != 'All':
    filtered = filtered[filtered['Conservation Status'] == selected_status]
if selected_country != 'All':
    filtered = filtered[filtered['Country/Region'] == selected_country]
if year_range and 'Year' in filtered:
    filtered = filtered[(filtered['Year'] >= year_range[0]) & (filtered['Year'] <= year_range[1])]

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric('Observations', len(filtered))
col2.metric('Species', filtered['Common Name'].nunique())
col3.metric('Countries', filtered['Country/Region'].nunique())
if {'Observed Length (m)','Observed Weight (kg)'} <= set(filtered.columns):
    corr_val = filtered['Observed Length (m)'].corr(filtered['Observed Weight (kg)'])
    col4.metric('Length-Weight Corr', f"{corr_val:.2f}" if pd.notnull(corr_val) else 'NA')
else:
    col4.metric('Length-Weight Corr', 'NA')

tab_overview, tab_length_weight, tab_geo, tab_temporal, tab_conservation, tab_eda_outputs = st.tabs([
    'Overview','Length vs Weight','Geographic','Temporal','Conservation','EDA Outputs'])

with tab_overview:
    st.subheader('Species Distribution')
    species_counts = filtered['Common Name'].value_counts().head(15)
    fig = px.bar(species_counts, x=species_counts.index, y=species_counts.values, labels={'x':'Species','y':'Count'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Habitat Type (Top 10)')
    if 'Habitat Type' in filtered.columns:
        habitat_counts = filtered['Habitat Type'].value_counts().head(10)
        fig2 = px.bar(habitat_counts, x=habitat_counts.values, y=habitat_counts.index, orientation='h', labels={'x':'Count','y':'Habitat'})
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander('View Filtered DataFrame'):
        st.dataframe(filtered)

with tab_length_weight:
    st.subheader('Length vs Weight')
    if {'Observed Length (m)','Observed Weight (kg)'} <= set(filtered.columns):
        fig = px.scatter(filtered, x='Observed Length (m)', y='Observed Weight (kg)', color='Common Name', opacity=0.65,
                         hover_name='Common Name', trendline='ols')
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            hist1 = px.histogram(filtered, x='Observed Length (m)', nbins=30, title='Length Distribution')
            st.plotly_chart(hist1, use_container_width=True)
        with c2:
            hist2 = px.histogram(filtered, x='Observed Weight (kg)', nbins=30, title='Weight Distribution')
            st.plotly_chart(hist2, use_container_width=True)
    else:
        st.info('Length/Weight columns not available.')

with tab_geo:
    st.subheader('Top Countries/Regions')
    country_counts = filtered['Country/Region'].value_counts().head(12)
    fig = px.bar(country_counts, x=country_counts.values, y=country_counts.index, orientation='h', labels={'x':'Count','y':'Country/Region'})
    st.plotly_chart(fig, use_container_width=True)

    if 'Conservation Status' in filtered.columns:
        st.subheader('Country vs Conservation Status (Heatmap)')
        top_countries = filtered['Country/Region'].value_counts().head(8).index
        crosstab = pd.crosstab(filtered['Country/Region'].apply(lambda x: x if x in top_countries else 'Other'), filtered['Conservation Status'])
        heat = px.imshow(crosstab, text_auto=True, aspect='auto', color_continuous_scale='viridis')
        st.plotly_chart(heat, use_container_width=True)

with tab_temporal:
    st.subheader('Observations Over Time')
    if 'Year' in filtered.columns:
        yearly = filtered.groupby('Year').size().reset_index(name='Count')
        line_fig = px.line(yearly, x='Year', y='Count', markers=True)
        st.plotly_chart(line_fig, use_container_width=True)
    if 'Month' in filtered.columns:
        monthly = filtered.groupby('Month').size().reset_index(name='Count')
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        monthly['Month Name'] = monthly['Month'].apply(lambda x: month_names[int(x)-1] if pd.notnull(x) and 1 <= int(x) <= 12 else str(x))
        bar = px.bar(monthly, x='Month Name', y='Count', category_orders={'Month Name': month_names}, title='Seasonality')
        st.plotly_chart(bar, use_container_width=True)

with tab_conservation:
    st.subheader('Conservation Status Distribution')
    if 'Conservation Status' in filtered.columns:
        status_counts = filtered['Conservation Status'].value_counts()
        pie = px.pie(values=status_counts.values, names=status_counts.index, title='Conservation Status')
        st.plotly_chart(pie, use_container_width=True)
    if {'Observed Length (m)','Conservation Status'} <= set(filtered.columns):
        st.subheader('Length by Conservation Status')
        box = px.box(filtered, x='Conservation Status', y='Observed Length (m)', points='outliers')
        st.plotly_chart(box, use_container_width=True)

with tab_eda_outputs:
    st.subheader('EDA Outputs (Loaded from eda_outputs)')
    numeric_summary = load_optional_csv(EDA_DIR / 'numeric_summary.csv')
    correlation_matrix = load_optional_csv(EDA_DIR / 'correlation_matrix.csv')
    outliers = load_optional_csv(EDA_DIR / 'outliers_iqr.csv')
    regression_stats = load_json(EDA_DIR / 'length_weight_regression.json')
    dq = load_json(EDA_DIR / 'data_quality.json')

    if not numeric_summary.empty:
        st.markdown('**Numeric Summary**')
        st.dataframe(numeric_summary)
    if not correlation_matrix.empty:
        st.markdown('**Correlation Matrix**')
        st.dataframe(correlation_matrix)
    if not outliers.empty:
        st.markdown('**Outlier Report (IQR)**')
        st.dataframe(outliers)
    if regression_stats:
        st.markdown('**Length-Weight Regression Stats**')
        st.json(regression_stats)
    if dq:
        st.markdown('**Data Quality Summary**')
        st.json(dq)

    # Download buttons
    if not numeric_summary.empty:
        st.download_button('Download Numeric Summary CSV', numeric_summary.to_csv().encode('utf-8'), 'numeric_summary.csv', 'text/csv')
    if not outliers.empty:
        st.download_button('Download Outlier Report CSV', outliers.to_csv().encode('utf-8'), 'outliers_iqr.csv', 'text/csv')
    if regression_stats:
        st.download_button('Download Regression JSON', json.dumps(regression_stats, indent=2).encode('utf-8'), 'length_weight_regression.json', 'application/json')
    if dq:
        st.download_button('Download Data Quality JSON', json.dumps(dq, indent=2).encode('utf-8'), 'data_quality.json', 'application/json')

st.caption('Dashboard generated from crocodile dataset. Run EDA script first for enriched outputs.')
