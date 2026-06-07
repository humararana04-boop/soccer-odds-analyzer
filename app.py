import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

# Page Setup (Modern & Professional Dashboard)
st.set_page_config(page_title="AI Soccer Odds Pattern Detector", layout="wide")

st.title("⚽ AI Soccer Odds Analysis & Deceptive Pattern Detector")
st.write("Upload your raw XLSX/CSV file to automatically clean data, align the 4-hour pre-match timeline, and detect suspicious bookmaker movements.")

# --- SIDEBAR: CLIENT CONTROLS ---
st.sidebar.header("🎛️ Control Panel")

# 1. File Uploader (Zero-Installation for Client)
uploaded_file = st.sidebar.file_uploader("Upload Raw Odds File (XLSX or CSV)", type=["xlsx", "csv"])

# 2. Adaptive AI Parameters (Sliders for Client)
st.sidebar.subheader("🤖 AI Detector Tuning")
z_threshold = st.sidebar.slider(
    "AI Sensitivity Threshold (Z-Score)", 
    min_value=1.5, max_value=4.0, value=2.5, step=0.1,
    help="Lower values mean stricter detection (more flags). Higher values flag only extreme movements."
)

sig_move_pct = st.sidebar.slider(
    "Significant Move Trigger (%)", 
    min_value=1.0, max_value=10.0, value=5.0, step=0.5,
    help="The percentage change in odds to trigger the 'First Significant Move' timestamp."
) / 100.0

# --- CORE PROCESSING ENGINE ---
@st.cache_data
def process_data(file, z_thresh, sig_thresh):
    # Load File safely
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
        
    # Convert timestamps
    df['kickoff_time_utc'] = pd.to_datetime(df['kickoff_time_utc'])
    df['entry_time_utc'] = pd.to_datetime(df['entry_time_utc'])
    
    # Calculate minutes before kickoff
    df['mins_before_kickoff'] = (df['kickoff_time_utc'] - df['entry_time_utc']).dt.total_seconds() / 60
    
    # Filter for the pure 4-hour pre-match window (0 to 240 mins)
    cleaned_df = df[(df['mins_before_kickoff'] >= 0) & (df['mins_before_kickoff'] <= 240)].copy()
    
    if cleaned_df.empty:
        return None, None, None
        
    # Sort data chronologically for timeline analysis
    cleaned_df = cleaned_df.sort_values(by=['fixture_id', 'sportsbook', 'market', 'selection', 'mins_before_kickoff'], ascending=[True, True, True, True, False])
    
    # 1. AI Anomaly/Deceptive Pattern Detection
    cleaned_df['odds_pct_change'] = cleaned_df.groupby(['fixture_id', 'sportsbook', 'market', 'selection'])['odds_decimal'].pct_change().fillna(0)
    
    # Compare each provider against the overall market average for that specific selection
    market_mean = cleaned_df.groupby(['fixture_id', 'market', 'selection'])['odds_pct_change'].transform('mean')
    market_std = cleaned_df.groupby(['fixture_id', 'market', 'selection'])['odds_pct_change'].transform('std').fillna(0.01)
    
    # Handle division by zero or tiny std
    market_std = np.where(market_std == 0, 0.01, market_std)
    
    cleaned_df['z_score'] = (cleaned_df['odds_pct_change'] - market_mean) / market_std
    cleaned_df['is_suspicious'] = np.where(abs(cleaned_df['z_score']) > z_thresh, "SUSPICIOUS / DECEPTIVE", "ROUTINE")
    
    # 2. Metrics Generation
    metrics_list = []
    grouped = cleaned_df.groupby(['fixture_id', 'sportsbook', 'market', 'selection'])
    
    for name, group in grouped:
        fix_id, book, mkt, sel = name
        odds = group['odds_decimal'].values
        times = group['mins_before_kickoff'].values
        
        initial_odds = odds[0]
        final_odds = odds[-1]
        max_swing = np.max(odds) - np.min(odds)
        mean_change = np.mean(np.abs(group['odds_pct_change']))
        
        # Detect First Significant Move
        sig_move_time = "No Significant Move"
        for i in range(1, len(odds)):
            if abs(odds[i] - odds[i-1]) / odds[i-1] >= sig_thresh:
                sig_move_time = f"{int(times[i])} mins before KO"
                break
                
        metrics_list.append({
            'Fixture ID': fix_id,
            'League': group['league'].iloc[0],
            'Match': f"{group['home_team'].iloc[0]} vs {group['away_team'].iloc[0]}",
            'Provider (Sportsbook)': book,
            'Market': mkt,
            'Selection': sel,
            'Initial Odds': initial_odds,
            'Final Odds': final_odds,
            'Max Swing': round(max_swing, 3),
            'Mean Absolute Change': round(mean_change, 4),
            'Time of First Sig Move': sig_move_time,
            'Suspicious Fluctuations Found': int((group['is_suspicious'] == "SUSPICIOUS / DECEPTIVE").sum())
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    
    # 3. Regional / League Deviation Analysis
    league_deviations = metrics_df.groupby('League').agg(
        Avg_Max_Swing=('Max Swing', 'mean'),
        Total_Suspicious_Flags=('Suspicious Fluctuations Found', 'sum')
    ).reset_index()
    
    global_avg_swing = metrics_df['Max Swing'].mean()
    league_deviations['Deviation From Global Avg'] = league_deviations['Avg_Max_Swing'] - global_avg_swing
    league_deviations = league_deviations.sort_values(by='Total_Suspicious_Flags', ascending=False)
    
    return cleaned_df, metrics_df, league_deviations

# --- DASHBOARD VISUALS & TABS ---
if uploaded_file is not None:
    with st.spinner("Processing data and applying AI models... Please wait."):
        cleaned_data, summary_metrics, league_analysis = process_data(uploaded_file, z_threshold, sig_move_pct)
        
    if cleaned_data is not None:
        st.success("✅ System Processed Data Successfully!")
        
        # Layout Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary Metrics", "📈 Live Odds Visualizer", "🌍 League Deviations", "📋 Methodology & Download"])
        
        # TAB 1: SUMMARY METRICS
        with tab1:
            st.subheader("Key Performance & Price Tracking Metrics")
            st.dataframe(summary_metrics, use_container_width=True)
            
        # TAB 2: INTERACTIVE VISUALIZATION
        with tab2:
            st.subheader("Minute-by-Minute Timeline Chart")
            available_matches = cleaned_data['fixture_id'].unique()
            selected_fix = st.selectbox("Select a Match / Fixture ID:", available_matches)
            
            match_df = cleaned_data[cleaned_data['fixture_id'] == selected_fix]
            st.write(f"**Match:** {match_df['home_team'].iloc[0]} vs {match_df['away_team'].iloc[0]} | **League:** {match_df['league'].iloc[0]}")
            
            available_selections = match_df['selection'].unique()
            selected_sel = st.selectbox("Select Market Selection:", available_selections)
            
            chart_df = match_df[match_df['selection'] == selected_sel].sort_values('mins_before_kickoff', ascending=False)
            
            # Interactive Line Graph
            fig = px.line(
                chart_df, 
                x="mins_before_kickoff", 
                y="odds_decimal", 
                color="sportsbook",
                symbol="is_suspicious",
                title=f"Odds Fluctuations inside 4-Hour Window (Selection: {selected_sel})",
                labels={"mins_before_kickoff": "Minutes Before Kickoff", "odds_decimal": "Odds (Decimal)"}
            )
            fig.update_xaxes(autorange="reverse") # Show 240 mins going down to 0 mins
            st.plotly_chart(fig, use_container_width=True)
            
        # TAB 3: LEAGUE DEVIATIONS
        with tab3:
            st.subheader("Leagues/Regions Deviating Markedly From Global Average")
            st.write("This table highlights leagues with high volatility or a high volume of suspicious AI flags.")
            st.dataframe(league_analysis, use_container_width=True)
            
            fig_bar = px.bar(
                league_analysis.head(15), 
                x='League', 
                y='Total_Suspicious_Flags', 
                title="Top 15 Most Volatile / Flagged Soccer Leagues",
                color='Avg_Max_Swing'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # TAB 4: METHODOLOGY & EXPORT WORKBOOK
        with tab4:
            st.subheader("📥 Download Refreshed Workbook")
            st.write("Click the button below to download the fully annotated data containing all custom metrics and AI flags for your offline use.")
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                cleaned_data.to_excel(writer, sheet_name="Cleaned & AI Flagged", index=False)
                summary_metrics.to_excel(writer, sheet_name="Summary Metrics", index=False)
                league_analysis.to_excel(writer, sheet_name="League Deviations", index=False)
            
            st.download_button(
                label="🚀 Download Refreshed XLSX Workbook",
                data=buffer.getvalue(),
                file_name="Refreshed_AI_Odds_Analysis.xlsx",
                mime="application/vnd.ms-excel"
            )
            
            st.markdown("""
            ---
            ### 📝 Methodology & Recalibration Note
            * **Timeline Alignment:** The system filters out any data outside the **4-hour pre-match window (-240 to 0 minutes)** based on the difference between `kickoff_time_utc` and `entry_time_utc`.
            * **Deceptive Pattern Identification:** Rather than static rules, an adaptive statistical **Z-score mechanism** monitors how fast a specific provider shifts its line relative to the real-time average shifting behavior of all providers on that selection. 
            * **Model Recalibration:** When a provider's movement deviates beyond the user-defined **AI Sensitivity Threshold**, it is labeled as `SUSPICIOUS / DECEPTIVE`. To recalibrate the system after cleaning out noise, simply adjust the **Sensitivity Slider** in the sidebar. Increasing the threshold filters out routine market corrections and captures only extreme market shocks.
            """)
else:
    st.info("💡 Awaiting file upload. Please drop your CSV or XLSX file in the sidebar to start the system.")
