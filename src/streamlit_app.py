from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Add src to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))


class DatabaseManager:
    """Handle all database operations"""

    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv('MYSQL_HOST'),
                database=os.getenv('MYSQL_DB'),
                user=os.getenv('MYSQL_USER'),
                password=os.getenv('MYSQL_PASS'),
                port=os.getenv('MYSQL_PORT')
            )
            if self.connection.is_connected():
                st.success("✅ Database connected successfully")
        except Error as e:
            st.error(f"❌ Database connection failed: {e}")
            self.connection = None

    def execute_query(self, query):
        """Execute SQL query and return DataFrame"""
        if not self.connection or not self.connection.is_connected():
            self.connect()

        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()
            return pd.DataFrame(data, columns=columns)
        except Error as e:
            st.error(f"❌ Query execution failed: {e}")
            return pd.DataFrame()

    def get_cripd_data(self):
        """Fetch statistical model data with date formatting"""
        query = """
        SELECT 
            comp_id,
            CONCAT(yy, '-', LPAD(mm, 2, '0'), '-', LPAD(dd, 2, '0')) as yyyymmdd,
            yy, mm, dd,
            pd_1, pd_2, pd_3, pd_6, pd_12, pd_24, pd_36, pd_60,
            poe_1, poe_2, poe_3, poe_6, poe_12, poe_24, poe_36, poe_60,
            task_date
        FROM mlops_pd.cripd_daily
        ORDER BY comp_id, yy, mm, dd
        """
        return self.execute_query(query)

    def get_mlpd_data(self):
        """Fetch ML model data"""
        query = """
        SELECT 
            comp_id,
            yyyymmdd,
            yyyy, mm,
            pd_1, poe_1, econ,
            task_date
        FROM mlops_pd.mlpd_daily_dev
        ORDER BY comp_id, yyyymmdd
        """
        return self.execute_query(query)

    def get_ground_truth_data(self):
        """Fetch ground truth data"""
        query = """
        SELECT 
            comp_id,
            yyyymmdd,
            event_type,
            task_date
        FROM mlops_pd.pd_ground_truth
        ORDER BY comp_id, yyyymmdd
        """
        return self.execute_query(query)

    def get_ground_truth_event(self, task_date: str):
        """Fetch ground truth data"""
        query = f"""
        SELECT gt.comp_id,
		gt.yyyymmdd,
		gt.event_type,
		gt.task_date
        FROM mlops_pd.pd_ground_truth as gt
        where gt.task_date = {task_date}
        ORDER BY gt.comp_id, yyyymmdd desc;
        """
        return self.execute_query(query)

    def get_result_table(self):
        """Fetch ground truth data"""
        query = f"""
        SELECT cd.comp_id, CONCAT(LPAD(cd.yy, 4, '20'), '-', LPAD(cd.mm, 2, '0'), '-', LPAD(cd.dd, 2, '0')) AS yyyy_mm_dd, cd.pd_1 as stats_pd_1, cd.poe_1 as stats_poe_1, md.pd_1 as ml_pd_1, md.poe_1 as ml_poe_1, cd.task_date, pd.event_type
        FROM mlops_pd.cripd_daily as cd
        LEFT JOIN mlops_pd.mlpd_daily_dev as md
            ON cd.comp_id = md.comp_id AND
            CONCAT(LPAD(cd.yy, 4, '20'), '-', LPAD(cd.mm, 2, '0'), '-', LPAD(cd.dd, 2, '0')) = md.yyyymmdd AND
            cd.task_date = md.task_date
        LEFT JOIN mlops_pd.pd_ground_truth as pd   
            ON cd.comp_id = pd.comp_id AND
            CONCAT(LPAD(cd.yy, 4, '20'), '-', LPAD(cd.mm, 2, '0'), '-', LPAD(cd.dd, 2, '0')) = pd.yyyymmdd AND
            cd.task_date = pd.task_date;
        """
        return self.execute_query(query)

    def get_unique_companies(self):
        """Get list of unique company numbers"""
        query = """
        SELECT DISTINCT comp_id 
        FROM mlops_pd.cripd_daily 
        ORDER BY comp_id
        """
        df = self.execute_query(query)
        return df['comp_id'].tolist() if not df.empty else []

    def get_country_metrics(self):
        query = """
        SELECT 
            e.econ_name,
            mt.metrics_name,
            m.value
        FROM 
            mlops_pd.econ e
        JOIN 
            mlops_pd.model mo ON e.econ_id = mo.econ_id
        JOIN 
            mlops_pd.metrics m ON mo.model_id = m.model_id
        JOIN 
            mlops_pd.metrics_type mt ON m.metrics_type_id = mt.metrics_type_id;
        """
        df = self.execute_query(query)
        return df
    
    def get_econ_list(self):
        df = self.execute_query("SELECT econ_id, econ_name FROM mlops_pd.econ;")
        return df

    def close_connection(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()


class DataProcessor:
    """Process and transform data for dashboard"""

    @staticmethod
    def calculate_forecast_date(yyyymmdd_str):
        """Calculate forecast date (next month)"""
        try:
            current_date = datetime.strptime(yyyymmdd_str, '%Y-%m-%d')
            # Add one month
            if current_date.month == 12:
                forecast_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                forecast_date = current_date.replace(month=current_date.month + 1, day=current_date.day)
            return forecast_date.strftime('%Y-%m-%d')
        except:
            return "N/A"

    @staticmethod
    def calculate_past_date(yyyymmdd_str):
        """Calculate past date (one month before)"""
        try:
            current_date = datetime.strptime(yyyymmdd_str, '%Y-%m-%d')
            # Subtract one month
            if current_date.month == 1:
                past_date = current_date.replace(year=current_date.year - 1, month=12)
            else:
                past_date = current_date.replace(month=current_date.month - 1)
            return past_date.strftime('%Y-%m-%d')
        except:
            return "N/A"

    @staticmethod
    def merge_prediction_data(cripd_df, mlpd_df, ground_truth_df):
        """Merge all prediction data properly - one row per date"""
        # Prepare statistical data
        stats_data = cripd_df[['comp_id', 'yyyymmdd', 'pd_1', 'poe_1', 'task_date']].copy()
        stats_data.columns = ['comp_id', 'yyyymmdd', 'stats_pd_1', 'stats_poe_1', 'task_date']

        # Prepare ML data
        ml_data = mlpd_df[['comp_id', 'yyyymmdd', 'pd_1', 'poe_1', 'task_date']].copy()
        ml_data.columns = ['comp_id', 'yyyymmdd', 'ml_pd_1', 'ml_poe_1', 'task_date']

        # Merge stats and ML data on comp_id, yyyymmdd, and task_date
        merged_df = pd.merge(stats_data, ml_data, on=['comp_id', 'yyyymmdd', 'task_date'], how='outer')

        # Merge ground truth data
        merged_df = pd.merge(merged_df, ground_truth_df, on=['comp_id', 'yyyymmdd', 'task_date'], how='left')

        # Calculate forecast date
        merged_df['forecast_date'] = merged_df['yyyymmdd'].apply(DataProcessor.calculate_forecast_date)

        # Sort by company and date
        merged_df = merged_df.sort_values(['comp_id', 'yyyymmdd'])

        return merged_df

    @staticmethod
    def get_event_type_description(event_type):
        """Convert event type number to description"""
        if pd.isna(event_type) or event_type == "NaN":
            return "NaN"

        event_mapping = {
            0: "No Default",
            1: "Default",
            2: "Exit the Market"
        }
        return event_mapping.get(int(event_type), "Unknown")

    @staticmethod
    def get_current_and_previous_events(ground_truth_df, comp_id, task_date):
        """Get current and previous event information from ground truth table"""
        company_gt = ground_truth_df[(ground_truth_df['comp_id'] == comp_id) &
                                     (ground_truth_df['task_date'] == task_date)].copy()

        # print(company_gt)

        if company_gt.empty:
            return None, None, None, None

        # Sort by date descending to get most recent first
        company_gt = company_gt.sort_values('yyyymmdd', ascending=False)

        # Get current (most recent) event
        current_event = company_gt.iloc[0]
        current_date = current_event['yyyymmdd']
        current_event_type = current_event['event_type']

        # Get previous event (second most recent)
        if len(company_gt) > 1:
            previous_event = company_gt.iloc[1]
            previous_date = previous_event['yyyymmdd']
            previous_event_type = previous_event['event_type']
        else:
            previous_date = "NaN"
            previous_event_type = "NaN"

        return current_date, current_event_type, previous_date, previous_event_type


# Load company mapping
@st.cache_data
def load_company_info():
    """Load company mapping data"""
    try:
        company_mapping_path = Path("./src/company_info.csv")
        if company_mapping_path.exists():
            return pd.read_csv(company_mapping_path)
        else:
            st.warning("Company mapping file not found")
            return None
    except Exception as e:
        st.error(f"Could not load company mapping: {e}")
        return None


class DashboardUI:
    """Handle UI components and visualizations"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.data_processor = DataProcessor()

    def render_home_page(self):
        """Render home page with overview"""
        st.title("🏦 Probability of Default Prediction Dashboard")

        st.markdown("""
        ## Welcome to the PD Prediction System

        This dashboard provides comprehensive analysis of probability of default (PD) and 
        probability of exit (POE) predictions using both statistical and machine learning models.

        ### Key Features:
        - **Statistical Model**: Traditional statistical approach for PD/POE prediction
        - **ML Model**: Advanced machine learning model for enhanced accuracy
        - **Ground Truth Comparison**: Compare predictions with actual outcomes
        - **Real-time Data**: Dynamic data updates from MySQL database

        ### Navigation:
        - **Home**: Overview and system information
        - **Individual Predictions**: Detailed company-specific analysis
        """)

        # Display system overview
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📊 Data Sources", "3 Tables")
            st.caption("CRIPD Daily, MLPD Daily, Ground Truth")

        with col2:
            companies = self.db_manager.get_unique_companies()
            st.metric("🏢 Companies", len(companies))
            st.caption("Active companies in system")

        with col3:
            st.metric("🔄 Update Frequency", "Daily")
            st.caption("Real-time data refresh")

    def render_individual_predictions(self):
        """Render individual company predictions page"""
        st.title("🎯 Individual Company Predictions")

        # Company selection
        companies = self.db_manager.get_unique_companies()
        companies = [int(c) for c in companies]
        if not companies:
            st.error("❌ No companies found in database")
            return

        company_info = load_company_info()
        if company_info is None:
            st.error("Company info not available")
            return

        company_info['U3_COMPANY_NUMBER'] = company_info['U3_COMPANY_NUMBER'].astype(int)
        filtered_df = company_info[company_info['U3_COMPANY_NUMBER'].isin(companies)]
        company_info_df = filtered_df[
            ['ID_BB_UNIQUE', 'U3_COMPANY_NUMBER', 'Company_name', 'ticker', 'INDUSTRY_SUBGROUP']].copy()

        # Company search
        company_search = st.text_input("🔍 Search Company", placeholder="Enter company name or ID")

        if company_search:
            # Filter companies
            filtered_companies = company_info_df[
                company_info_df['ID_BB_UNIQUE'].str.contains(company_search, case=False, na=False) |
                company_info_df['Company_name'].str.contains(company_search, case=False, na=False) |
                company_info_df['ticker'].str.contains(company_search, case=False, na=False)
                ]
        else:
            filtered_companies = company_info_df

        if len(filtered_companies) > 0:
            company_options = [
                f"{row['ticker']} ({row['ID_BB_UNIQUE']}) ({row['U3_COMPANY_NUMBER']})"
                for _, row in filtered_companies.iterrows()
            ]

            selected_company = st.selectbox(
                "Select Company",
                company_options,
                index=0
            )

            # Extract company ID
            company_no = str(selected_company.split("(")[-1].split(")")[0])
            # Filter the row for the selected company
            selected_row = company_info_df[company_info_df['U3_COMPANY_NUMBER'] == int(company_no)]

            # Extract values safely
            if not selected_row.empty:
                company_name = selected_row.iloc[0]['Company_name']
                industry_subgroup = selected_row.iloc[0]['INDUSTRY_SUBGROUP']
            else:
                company_name = None
                industry_subgroup = None

        if st.button("🔍 Analyze Company", type="primary"):
            self.display_company_analysis(int(company_no), company_name, industry_subgroup)

    def display_company_analysis(self, comp_id, company_name, industry_subgroup):
        """Display comprehensive company analysis"""
        st.subheader(f"📈 Analysis for Company: {company_name},  {industry_subgroup}")

        # Fetch data
        cripd_df = self.db_manager.get_cripd_data()
        mlpd_df = self.db_manager.get_mlpd_data()
        ground_truth_df = self.db_manager.get_ground_truth_data()

        cripd_df['yyyymmdd'] = pd.to_datetime(cripd_df['yyyymmdd']).dt.strftime('%Y-%m-%d')
        mlpd_df['yyyymmdd'] = pd.to_datetime(mlpd_df['yyyymmdd']).dt.strftime('%Y-%m-%d')
        ground_truth_df['yyyymmdd'] = pd.to_datetime(ground_truth_df['yyyymmdd']).dt.strftime('%Y-%m-%d')

        # Filter for selected company
        company_cripd = cripd_df[cripd_df['comp_id'] == comp_id]
        company_mlpd = mlpd_df[mlpd_df['comp_id'] == comp_id]
        company_gt = ground_truth_df[ground_truth_df['comp_id'] == comp_id]

        if company_cripd.empty and company_mlpd.empty:
            st.warning(f"⚠️ No data found for company {comp_id}")
            return

        # Merge company data
        merged_data = self.data_processor.merge_prediction_data(
            company_cripd, company_mlpd, company_gt
        )

        # Display latest prediction information
        self.display_latest_prediction_info(merged_data, comp_id)

        # Display historical plots
        self.display_historical_plots(merged_data, comp_id)

    def display_latest_prediction_info(self, merged_data, comp_id):
        """Display latest prediction information with proper event handling"""
        if merged_data.empty:
            return

        # Get latest record with actual data
        latest_record = merged_data.iloc[-1]

        # Get ground truth data for event information
        ground_truth_df = self.db_manager.get_ground_truth_data()
        # ground_truth_df['yyyymmdd'] = pd.to_datetime(ground_truth_df['yyyymmdd']).dt.strftime('%Y-%m-%d')
        current_date, current_event_type, previous_date, previous_event_type = self.data_processor.get_current_and_previous_events(
            ground_truth_df, comp_id, latest_record['task_date'])

        st.markdown("### Latest Prediction Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📅 Current Date:",
                      latest_record['yyyymmdd'])
            # if previous_date != "NaN" and previous_date is not None:
            #     previous_event_desc = self.data_processor.get_event_type_description(previous_event_type)
            #     # st.metric("Previous Event Type", previous_event_desc)
            #     # st.caption(f"Date: {previous_date}")
            # else:
            #     st.metric("Previous Event Type", "NaN")
            #     st.caption("No previous event data available")

        with col2:
            st.metric("📅 Forecast Date:",
                      latest_record['forecast_date'])

            if current_date is not None:
                current_event_desc = self.data_processor.get_event_type_description(current_event_type)
                st.metric("Current Event Type", current_event_desc)
                st.caption(f"Date: {current_date}")
            else:
                st.metric("Current Event Type", "N/A")
                st.caption("No ground truth data available")

        with col3:
            st.markdown("### 📋 Latest Predictions")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Statistical PD_1:** {latest_record.get('stats_pd_1', 'N/A'):.6f}")
                st.success(f"**Statistical POE_1:** {latest_record.get('stats_poe_1', 'N/A'):.6f}")

            with col2:
                st.warning(f"**ML PD_1:** {latest_record.get('ml_pd_1', 'N/A'):.6f}")
                st.warning(f"**ML POE_1:** {latest_record.get('ml_poe_1', 'N/A'):.6f}")

    def display_historical_plots(self, merged_data, comp_id):
        """Display historical trend plots"""
        if merged_data.empty:
            return

        st.markdown("### 📈 Historical Trends")

        # Prepare data for plotting
        plot_data = merged_data.copy()
        plot_data['yyyymmdd'] = pd.to_datetime(plot_data['yyyymmdd'])
        plot_data = plot_data.sort_values('yyyymmdd')

        # Create PD plot
        fig_pd = go.Figure()

        # Add statistical PD line
        if 'stats_pd_1' in plot_data.columns:
            fig_pd.add_trace(go.Scatter(
                x=plot_data['yyyymmdd'],
                y=plot_data['stats_pd_1'],
                mode='lines+markers',
                name='Statistical PD_1',
                line=dict(color='blue', width=2)
            ))

        # Add ML PD line
        if 'ml_pd_1' in plot_data.columns:
            fig_pd.add_trace(go.Scatter(
                x=plot_data['yyyymmdd'],
                y=plot_data['ml_pd_1'],
                mode='lines+markers',
                name='ML PD_1',
                line=dict(color='red', width=2)
            ))

        fig_pd.update_layout(
            title=f'Probability of Default (PD_1) - Company {comp_id}',
            xaxis_title='Date',
            yaxis_title='Probability',
            hovermode='x unified'
        )

        st.plotly_chart(fig_pd, use_container_width=True)

        # Create POE plot
        fig_poe = go.Figure()

        # Add statistical POE line
        if 'stats_poe_1' in plot_data.columns:
            fig_poe.add_trace(go.Scatter(
                x=plot_data['yyyymmdd'],
                y=plot_data['stats_poe_1'],
                mode='lines+markers',
                name='Statistical POE_1',
                line=dict(color='green', width=2)
            ))

        # Add ML POE line
        if 'ml_poe_1' in plot_data.columns:
            fig_poe.add_trace(go.Scatter(
                x=plot_data['yyyymmdd'],
                y=plot_data['ml_poe_1'],
                mode='lines+markers',
                name='ML POE_1',
                line=dict(color='orange', width=2)
            ))

        fig_poe.update_layout(
            title=f'Probability of Exit (POE_1) - Company {comp_id}',
            xaxis_title='Date',
            yaxis_title='Probability',
            hovermode='x unified'
        )

        st.plotly_chart(fig_poe, use_container_width=True)

        # Display data table
        st.markdown("### 📋 Historical Data")

        # Prepare display data with proper formatting
        display_data = plot_data[['yyyymmdd', 'stats_pd_1', 'stats_poe_1', 'ml_pd_1', 'ml_poe_1', 'event_type']].copy()

        # Add event type description
        display_data['Event_Description'] = display_data['event_type'].apply(
            lambda x: self.data_processor.get_event_type_description(x)
        )

        # Format the dataframe for better display
        display_data = display_data[
            ['yyyymmdd', 'stats_pd_1', 'stats_poe_1', 'ml_pd_1', 'ml_poe_1', 'event_type', 'Event_Description']]

        st.dataframe(display_data, use_container_width=True)

    def render_metrics_by_country(self):
        """Render Metrics by Country page"""
        st.title("🗺️📊 Metrics by Country")

        # Fetch all data for metrics by country

        metrics_by_country = self.db_manager.get_country_metrics()

        if metrics_by_country.empty:
            st.error("❌ No data found.")
            return

        # Fetch unique values for econ_name (Country) and metrics_name (Metric)
        countries = metrics_by_country['econ_name'].unique()
        metrics = metrics_by_country['metrics_name'].unique()

        # Dropdown for selecting country
        selected_country = st.selectbox("Select Country", options=countries)

        # Dropdown for selecting metric name
        selected_metric = st.selectbox("Select Metric", options=metrics)

        # Filter data based on the selected country and metric
        filtered_df = metrics_by_country[
            (metrics_by_country['econ_name'] == selected_country) & (
                        metrics_by_country['metrics_name'] == selected_metric)
            ]

        # Display filtered data
        if not filtered_df.empty:
            st.dataframe(filtered_df)
        else:
            st.warning(f"No data found for {selected_country} and {selected_metric}.")

    '''def display_model_performance(self):
        """Display Model Performance Monitoring section with country + 1M / 12M horizons."""
        st.title("📊 Model Performance Monitoring")

        # Country selector
        econ_df = self.db_manager.execute_query("SELECT econ_id, econ_name FROM mlops_pd.econ;")
        # assume execute_query returns a DataFrame with econ_id,econ_name
        econ_options = econ_df['econ_name'].tolist()
        econ_map = dict(zip(econ_df['econ_name'], econ_df['econ_id']))
        selected_country_name = st.selectbox("Select Country", options=econ_options)
        selected_econ_id = econ_map[selected_country_name]

        st.markdown(f"Showing results for **{selected_country_name}**")

        # Tab horizons: 1M (horizon=1) vs 12M (horizon=12)
        tab1, tab2 = st.tabs(["📅 1M Horizon", "📅 12M Horizon"])

        with tab1:
            self.render_model_comparison_horizon(
                horizon = 1,
                selected_econ = selected_econ_id
            )

        with tab2:
            self.render_model_comparison_horizon(
                horizon = 12,
                selected_econ = selected_econ_id
            )
'''
    
    def display_model_performance(self):
        st.title("📊 Model Performance Monitoring")

        # === Step 1: Get all countries from database ===
        econ_df = self.db_manager.execute_query("SELECT econ_id, econ_name FROM mlops_pd.econ;")
        econ_map = dict(zip(econ_df['econ_id'], econ_df['econ_name']))
        reverse_econ_map = {v: k for k, v in econ_map.items()}

        # === Step 2: Get econ_ids from LightGBM results ===
        query = """
            SELECT DISTINCT econ FROM mlops_pd.mlpd_backtesting_results
            WHERE horizon IN (1, 12)
            AND (auc_score IS NOT NULL OR pr_auc IS NOT NULL)
        """
        ml_econ_df = self.db_manager.execute_query(query)
        ml_econ_ids = set(ml_econ_df['econ'].dropna().astype(int).tolist())

        # === Step 3: Check for existing CSVs for stats model ===
        csv_dir = "data/cripd_metrics"
        csv_econ_ids = set()
        for econ_id in econ_df['econ_id']:
            path_1m = os.path.join(csv_dir, f"cripd_mly_1M_{econ_id}.csv")
            path_12m = os.path.join(csv_dir, f"cripd_yearly_12M_{econ_id}.csv")
            if os.path.exists(path_1m) or os.path.exists(path_12m):
                csv_econ_ids.add(econ_id)

        # === Step 4: Union of valid econ_ids ===
        valid_econ_ids = ml_econ_ids.union(csv_econ_ids)

        # === Step 5: Filter econ_df to only valid econ_ids ===
        econ_df_filtered = econ_df[econ_df['econ_id'].isin(valid_econ_ids)]
        econ_options = econ_df_filtered['econ_name'].tolist()
        econ_map_filtered = dict(zip(econ_df_filtered['econ_name'], econ_df_filtered['econ_id']))

        if not econ_options:
            st.warning("No countries with available data (ML or Stats).")
            return

        # === Step 6: Dropdown ===
        selected_country_name = st.selectbox("🌍 Select Country", options=econ_options)
        selected_econ_id = econ_map_filtered[selected_country_name]

        # === Proceed to tabs with selected country ===
        tab1, tab2 = st.tabs(["📅 1M Horizon", "📅 12M Horizon"])
        with tab1:
            self.render_model_comparison_horizon(horizon=1, selected_econ=selected_econ_id)

        with tab2:
            self.render_model_comparison_horizon(horizon=12, selected_econ=selected_econ_id)
    
    
    def render_model_comparison_horizon(self, horizon:int, selected_econ:int):
        """
        For a given horizon (1 or 12) and econ_id, load CRIPD stats from CSV and ML from DB,
        then plot comparison.
        """

        # --- Load statistical (CRIPD) data from CSV --- #
        # Construct filename depending on econ & horizon
        if horizon == 1:
            # CSV for 1M
            csv_stats_path = f"cripd_metrics/cripd_mly_1M_{selected_econ}.csv"         #set path as required
        elif horizon == 12:
            csv_stats_path = f"cripd_metrics/cripd_yearly_12M_{selected_econ}.csv"
        else:
            st.error(f"Horizon {horizon} not supported")
            return

        try:
            df_stats = pd.read_csv(csv_stats_path)
        except FileNotFoundError:
            st.error(f"Stats CSV not found: {csv_stats_path}")
            return

        # Ensure date parsing
        df_stats['date'] = pd.to_datetime(df_stats['date'])

        # --- Load ML (LightGBM) data from MySQL --- #
        # Query ML table
        query_ml = f"""
            SELECT date, auc_score as auc_score_ml, pr_auc as pr_auc_ml, test_positives
            FROM mlops_pd.mlpd_backtesting_results
            WHERE econ = {selected_econ}
              AND horizon = {horizon}
              AND (auc_score IS NOT NULL OR pr_auc IS NOT NULL)
            ORDER BY date asc
        """
        df_ml = self.db_manager.execute_query(query_ml)
        if df_ml.empty:
            st.warning("No ML data for this country / horizon.")
            return
        # parse date
        df_ml['date'] = pd.to_datetime(df_ml['date'])

        # --- Merge the two DataFrames on date --- #
        df = pd.merge(df_stats, df_ml, on='date', how='inner')

        if df.empty:
            st.warning("No overlapping dates between stats and ML data.")
            return

        # Rename stats columns to match suffix style or explicitly
        # Suppose df_stats has columns: date, auc_score, pr_auc
        df = df.rename(columns={
            'auc_score': 'auc_score_cripd',
            'pr_auc': 'pr_auc_cripd'
        })

        # Drop rows where both AUC values are NaN
        df_auc_cripd = df.dropna(subset=['auc_score_cripd'], how='all')
        df_auc_ml = df.dropna(subset=['auc_score_ml'], how='all')

        # --- Plotting --- #
        st.markdown("### 📈 AUC Score Comparison")
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Scatter(
            x=df_auc_cripd['date'], y=df_auc_cripd['auc_score_cripd'],
            mode='lines+markers', name='CRIPD Model AUC', line=dict(color='blue')
        ))
        fig_auc.add_trace(go.Scatter(
            x=df_auc_ml['date'], y=df_auc_ml['auc_score_ml'],
            mode='lines+markers', name='LightGBM Model AUC', line=dict(color='red')
        ))
        fig_auc.update_layout(title=f"AUC Score Over Time (Horizon = {horizon} month(s))",
                              xaxis_title='Date', yaxis_title='AUC', hovermode='x unified')
        st.plotly_chart(fig_auc, use_container_width=True)

        st.markdown("### 📉 PR AUC Score Comparison")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=df['date'], y=df['pr_auc_cripd'],
            mode='lines+markers', name='CRIPD Model PR AUC', line=dict(color='green')
        ))
        fig_pr.add_trace(go.Scatter(
            x=df['date'], y=df['pr_auc_ml'],
            mode='lines+markers', name='LightGBM Model PR AUC', line=dict(color='orange')
        ))
        fig_pr.update_layout(title=f"PR AUC Score Over Time (Horizon = {horizon} month(s))",
                             xaxis_title='Date', yaxis_title='PR AUC', hovermode='x unified')
        st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown("### 🧪 Test Positives Over Time (ML Model)")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df['date'], y=df['test_positives'],
            name='Test Positives', marker_color='crimson'
        ))
        fig_bar.update_layout(title=f"Test Positives Over Time (Horizon = {horizon} month(s))",
                              xaxis_title='Date', yaxis_title='Test Positives', hovermode='x unified')
        st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("📊 Summary Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"CRIPD AUC (Mean, {horizon}M)", f"{df['auc_score_cripd'].mean():.4f}")
                st.metric(f"CRIPD PR AUC (Mean, {horizon}M)", f"{df['pr_auc_cripd'].mean():.4f}")
            with col2:
                st.metric(f"ML AUC (Mean, {horizon}M)", f"{df['auc_score_ml'].mean():.4f}")
                st.metric(f"ML PR AUC (Mean, {horizon}M)", f"{df['pr_auc_ml'].mean():.4f}")
    

def main():
    """Main application function"""
    st.set_page_config(
        page_title="PD Prediction Dashboard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize database manager
    db_manager = DatabaseManager()

    # Initialize dashboard UI
    dashboard = DashboardUI(db_manager)

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Select Page:",
        ["🏠 Home", "🎯 Individual Predictions", "🗺️📊 Metrics by Country", "📊 Model Performance Monitoring"]
    )

    # Add refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Page routing
    if page == "🏠 Home":
        dashboard.render_home_page()
    elif page == "🎯 Individual Predictions":
        dashboard.render_individual_predictions()
    elif page == "🗺️📊 Metrics by Country":
        dashboard.render_metrics_by_country()
    elif page == "📊 Model Performance Monitoring":
        dashboard.display_model_performance()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔗 System Info**")
    st.sidebar.caption("Data updates daily")
    st.sidebar.caption("Real-time MySQL connection")

    # Close database connection when done
    if hasattr(db_manager, 'connection'):
        db_manager.close_connection()


if __name__ == "__main__":
    main()
