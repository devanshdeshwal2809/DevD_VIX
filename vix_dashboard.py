import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="VIX Forecasting Dashboard", layout="wide")
st.title("📊 VIX Forecasting Dashboard – India & USA")
st.markdown("Upload **VIX** CSVs and explore model performance with behavioral & other insights.")

# ⬇️ Added Tabs here
tab1, tab2, tab3 = st.tabs(["📊 Forecast & Models", "🌍Event Insights", "🧠Behavioral insights"])

# --- Note Block Style ---
note_block = """
<style>
.note-box {
    background-color: #1e1e1e;
    border-left: 6px solid #0d6efd;
    padding: 10px 15px;
    border-radius: 8px;
    margin-top: 10px;
    margin-bottom: 10px;
    font-size: 15px;
    color: #ffffff;
    border: 1px solid #444;
}
.metric-card {
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
    transition: 0.3s;
    border: 1px solid #333;
    box-shadow: 0 0 12px rgba(255,255,255,0.08);
}
.metric-card:hover {
    transform: scale(1.02);
}
.rmse-card {
    background: linear-gradient(135deg, #f44336, #ff7043);
    color: #fff;
}
.mae-card {
    background: linear-gradient(135deg, #ffeb3b, #fdd835);
    color: #333;
}
r2-card {
    background: linear-gradient(135deg, #4caf50, #66bb6a);
    color: #fff;
}
.metric-value {
    font-size: 28px;
    margin-top: 4px;
}
</style>
"""

note_template = lambda text: f"<div class='note-box'>📌 <b>Insight:</b> {text}</div>"

with tab1:
    st.markdown(note_block, unsafe_allow_html=True)

    
# --- Sidebar Uploads ---
    uploaded_files = st.sidebar.file_uploader("📂 Upload VIX CSV(s)", type="csv", accept_multiple_files=True)
    uploaded_index_file = st.sidebar.file_uploader("📈 Upload Index CSV (with Country column)", type="csv")
    uploaded_forecast_file = st.sidebar.file_uploader("📉 Upload Forecast CSV", type="csv")


# --- Load Data ---
if uploaded_files and uploaded_index_file and uploaded_forecast_file:
    index_df = pd.read_csv(uploaded_index_file, parse_dates=["Date"], dayfirst=True)
    index_df["Country"] = index_df["Country"].str.title()

    vix_dataframes = []
    for f in uploaded_files:
        df_temp = pd.read_csv(f, parse_dates=["Date"], dayfirst=True)
        df_temp["Country"] = df_temp["Country"].str.title()
        vix_dataframes.append(df_temp)
    combined_df = pd.concat(vix_dataframes, ignore_index=True)

    forecast_df = pd.read_csv(uploaded_forecast_file, parse_dates=["Date"])
    forecast_df["Country"] = forecast_df["Country"].str.title()

    # --- Filters ---
    country = st.sidebar.selectbox("🌍 Select Country", sorted(combined_df["Country"].dropna().astype(str).unique()))
    model = st.sidebar.selectbox("🧠 Select Model", sorted(combined_df["Model"].dropna().astype(str).unique()))

    df = combined_df[(combined_df["Country"] == country) & (combined_df["Model"] == model)].copy()
    index_country = index_df[index_df["Country"] == country].copy()
    df = pd.merge(df, index_country, on=["Date", "Country"], how="left", suffixes=("", "_Index"))
    df['Index'] = df['Index'].interpolate(method='linear')
    df = df.sort_values("Date")

    # --- Date Range Filter ---
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    start_date, end_date = st.slider("📅 Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

    st.download_button("⬇️ Download Filtered CSV", data=df.to_csv(index=False), file_name=f"{country}_{model}_Filtered.csv", mime="text/csv")

   # --- Metrics ---
    rmse = np.sqrt(mean_squared_error(df["Actual_VIX"], df["Predicted_VIX"]))
    mae = mean_absolute_error(df["Actual_VIX"], df["Predicted_VIX"])
    r2 = r2_score(df["Actual_VIX"], df["Predicted_VIX"])

    st.subheader("📌 Model Performance Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<div class='metric-card rmse-card'>📉 RMSE<br><div class='metric-value'>{rmse:.4f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card mae-card'>📊 MAE<br><div class='metric-value'>{mae:.4f}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card r2-card'>🧠 R² Score<br><div class='metric-value'>{r2:.4f}</div></div>", unsafe_allow_html=True)

    # --- Toggle Suggestion Section ---
    toggle = st.toggle("📊 Show Strategic Suggestions", value=True)
    if toggle:
        toggle_col1, toggle_col2 = st.columns(2)
        with toggle_col1:
            st.markdown("#### 📌 Recommendation Summary")
            band_stats = []
            for mdl in combined_df['Model'].unique():
                for ctry in combined_df['Country'].unique():
                    sub = combined_df[(combined_df['Model'] == mdl) & (combined_df['Country'] == ctry)].copy()
                    if not sub.empty:
                        sub['BandWidth'] = abs(sub['Predicted_VIX'] - sub['Actual_VIX']) * 2
                        band_stats.append({"Model": mdl, "Country": ctry, "Avg Band Width": sub['BandWidth'].mean()})
            df_band = pd.DataFrame(band_stats)
            preferred_model = df_band[df_band['Country'] == country].sort_values("Avg Band Width").iloc[0]
            st.success(f"✅ For {country}, the most stable model is **{preferred_model['Model']}** based on lowest band width.")
            st.markdown(note_template("This helps investors confidently choose the most reliable model for forecasting."), unsafe_allow_html=True)

        with toggle_col2:
            st.markdown("#### 🎯 Investor Action Summary")
            last_vix = df['Actual_VIX'].iloc[-1]
            if last_vix >= 25:
                trend = "High Volatility"
                advice = "📉 Market Volatile: Consider Hedging or Stay Defensive."
            elif last_vix >= 15:
                trend = "Moderate Volatility"
                advice = "🧐 Moderate Volatility: Monitor News and Earnings."
            else:
                trend = "Low Volatility"
                advice = "✅ Low Volatility: Ideal for Gradual Entry or Long-Term Positions."
            st.info(f"**{trend}** → {advice}")
            st.markdown(note_template("Based on current VIX levels, this summary guides strategic decisions."), unsafe_allow_html=True)

    # --- Forecast Chart ---
    forecast_filtered = forecast_df[(forecast_df["Country"] == country) & (forecast_df["Model"] == model)].copy()
    recent_actual = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=60)][["Date", "Actual_VIX"]].copy()

    chart1, chart2, chart3 = st.columns(3)
    with chart1:
        st.markdown("### 📈 Forecast Chart")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=recent_actual['Date'], y=recent_actual['Actual_VIX'], mode='lines', name='Actual VIX', line=dict(color='blue'), fill='tonexty'))
        fig1.add_trace(go.Scatter(x=forecast_filtered['Date'], y=forecast_filtered['Forecasted_VIX'], mode='lines', name='Forecasted VIX', line=dict(color='green', dash='dot'), fill='tonexty'))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(note_template("Forecast line shows future VIX compared to recent historical values."), unsafe_allow_html=True)

    with chart2:
        st.markdown("### 📊 Actual vs Predicted VIX ")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Actual_VIX'], mode='lines', name='Actual VIX', line=dict(color='royalblue'), fill='tonexty'))
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Predicted_VIX'], mode='lines', name='Predicted VIX', line=dict(color='tomato', dash='dot'), fill='tonexty'))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(note_template("Hybrid model closely tracks Actual VIX."), unsafe_allow_html=True)

    with chart3:
        st.markdown("### 📉 VIX vs Index")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['Date'], y=df['Actual_VIX'], name='VIX', line=dict(color='teal'), fill='tonexty', yaxis='y1'))
        fig3.add_trace(go.Scatter(x=df['Date'], y=df['Index'], name='Index', line=dict(color='orange'), fill='tonexty', yaxis='y2'))
        fig3.update_layout(yaxis=dict(title='VIX'), yaxis2=dict(title='Index', overlaying='y', side='right'))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(note_template("Helpful for investors to understand the behavior in market like overreaction or overconfidence."), unsafe_allow_html=True)

    # --- Correlation Matrix ---
    st.markdown("### 🔍 Deep Dive Analytics")
    deep1, deep2, deep3 = st.columns(3)

    with deep1:
        st.markdown("#### 🧠 Correlation Matrix")
        df_corr = df[['Actual_VIX', 'Predicted_VIX', 'Index']].copy()
        df_corr['Residual'] = df['Actual_VIX'] - df['Predicted_VIX']
        fig_corr = px.imshow(df_corr.corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown(note_template("+1 or near means strong correlation & residual near 0 means lesser error."), unsafe_allow_html=True)

    with deep2:
        st.markdown("#### 📉 Confidence Band")
        std_dev = np.std(df['Actual_VIX'] - df['Predicted_VIX'])
        df['Upper'] = df['Predicted_VIX'] + std_dev
        df['Lower'] = df['Predicted_VIX'] - std_dev
        fig_band = go.Figure()
        fig_band.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='Upper Band', line=dict(color='rgba(255,0,0,0.2)')))
        fig_band.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='Lower Band', fill='tonexty', line=dict(color='rgba(255,0,0,0.2)')))
        fig_band.add_trace(go.Scatter(x=df['Date'], y=df['Predicted_VIX'], name='Predicted', line=dict(color='orange')))
        st.plotly_chart(fig_band, use_container_width=True)
        st.markdown(note_template("Confidence Band gives a sense of uncertainty or prediction error."), unsafe_allow_html=True)

    with deep3:
        st.markdown("#### 🥧 Residual Distribution")
        residuals = df['Actual_VIX'] - df['Predicted_VIX']
        pie_data = pd.DataFrame({
            'Type': ['Overestimation', 'Underestimation', 'Perfect'],
            'Count': [(residuals < 0).sum(), (residuals > 0).sum(), (residuals == 0).sum()]
        })
        st.plotly_chart(px.pie(pie_data, names='Type', values='Count', color='Type',
            color_discrete_map={'Overestimation':'tomato','Underestimation':'dodgerblue','Perfect':'lightgreen'}), use_container_width=True)
        st.markdown(note_template("If over and underestimations are fairly balanced, the model might not be biased in one direction."), unsafe_allow_html=True)

    # --- Final Insights ---
    st.markdown("### 📊 Comparative Insights")
    final_col1, final_col2 = st.columns(2)

    with final_col1:
        st.markdown("#### 📏 Confidence Band Width Comparison")
        st.plotly_chart(px.bar(df_band, x="Model", y="Avg Band Width", color="Country", barmode="group"), use_container_width=True)
        st.markdown(note_template("Narrow Band relates to High confidence whereas Wider band relates to Lower confidence."), unsafe_allow_html=True)

    with final_col2:
        st.markdown("#### 🌍 Key Factors Affecting Volatility – India vs USA")
        factor_table = pd.DataFrame({
            'Factor': ['Inflation Data', 'Rate Decisions', 'Elections', 'Global Markets', 'Crude Oil Prices', 'USD/INR Movements', 'Geopolitical Events'],
            'India VIX Impact': ['Moderate increase', 'Spike if unexpected', 'Short-term spike', 'Moderate correlation', 'Impacts more', 'Direct impact', 'Spike'],
            'USA VIX Impact': ['Strong increase', 'Sharp spike (esp. Fed)', 'Volatile during uncertainty', 'Strong correlation', 'Minor', '-', 'Strong spike'],
            'Sentiment Effect': ['⚠️ Risk-off', '📉 Fear-driven', '🔁 Mixed', '🧠 Global appetite', '🇮🇳 Cost-push inflation', '🔺 FX flow impact', '😟 Panic hedge']
        })
        st.dataframe(factor_table, use_container_width=True)
        st.markdown(note_template("Understand how macro events affect investor behavior in India vs USA."), unsafe_allow_html=True)

#Tab 2 code (event insights part)
   
with tab2:
    st.subheader("🧠Event Insight Viewer")

    uploaded_event_file = st.file_uploader("📂 Upload Combined VIX + Index Dataset (2008–2024)", type="csv")

    if uploaded_event_file:
        df_event = pd.read_csv(uploaded_event_file, parse_dates=["Date"])
        df_event["Country"] = df_event["Country"].str.title()
        countries = df_event["Country"].unique()

        selected_country = st.selectbox("🌍 Select Country for Timeline View", countries)

        df_country = df_event[df_event["Country"] == selected_country].copy()
        df_country = df_country.sort_values("Date")
        df_country['Index'] = df_country['Index'].interpolate(method='linear')

        # Define event annotations for each country
        events_india = {
            '2008-09-15': "📉 2008 Global Crisis",
            '2013-05-22': "💸 Taper Tantrum",
            '2016-11-08': "💰 Demonetization",
            '2018-03-22': "🚧 Trade War",
            '2020-03-11': "🦠 COVID-19",
            '2021-06-01': "🔁 Recovery Begins",
            '2022-02-24': "⚔️ Russia-Ukraine",
            '2023-08-01': "📈 Rate Hike Fear",
            '2024-04-01': "🗳️ General Elections"
        }

        events_usa = {
            '2008-09-15': "📉 2008 Global Crisis",
            '2013-05-22': "💸 Taper Tantrum",
            '2016-11-08': "🗳️ US Elections",
            '2018-03-22': "🚧 Trade War",
            '2020-03-11': "🦠 COVID-19",
            '2021-06-01': "🔁 Recovery Begins",
            '2022-02-24': "⚔️ Russia-Ukraine",
            '2023-08-01': "📈 Inflation Scare",
            '2024-11-05': "🗳️ US Elections Upcoming"
        }

        events = events_india if selected_country == "India" else events_usa

        # ------------------------ Row 1: Timeline Viewer ------------------------
        st.markdown("### 📆 VIX & Index Timeline with Global Events")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_country['Date'], y=df_country['VIX'], mode='lines', name='VIX', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df_country['Date'], y=df_country['Index'], mode='lines', name='Index', line=dict(color='blue'), yaxis='y2'))

        for date_str, label in events.items():
            event_date = pd.to_datetime(date_str)
            if df_country['Date'].min() <= event_date <= df_country['Date'].max():
                fig.add_vline(x=event_date, line_width=1, line_dash="dash", line_color="gray")
                fig.add_annotation(x=event_date, y=max(df_country["VIX"]),
                                   text=label, showarrow=True, arrowhead=1, yshift=10,
                                   textangle=-90, bgcolor="black", font=dict(color="white", size=10))

        fig.update_layout(
            yaxis=dict(title='VIX'),
            yaxis2=dict(title='Index', overlaying='y', side='right'),
            legend=dict(orientation="h"),
            margin=dict(t=40, b=20),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # ----------------------- Row 2: Pre vs Post Event Sentiment -----------------------
        st.markdown("### 📊 Pre vs Post Event Sentiment Analysis")

        def classify_sentiment(vix_change, index_change):
            if vix_change > 0 and index_change < 0:
                return "😨 Panic"
            elif vix_change < 0 and index_change > 0:
                return "📈 Optimistic"
            elif vix_change > 0 and index_change > 0:
                return "😐 Mixed Reaction"
            elif vix_change < 0 and index_change < 0:
                return "📉 Uncertain"
            else:
                return "😐 Stable Mood"

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            pre_post_summary = []
            for event_date, label in events.items():
                event_day = pd.to_datetime(event_date)
                country_df = df_event[df_event["Country"] == selected_country].copy()

                pre_window = country_df[(country_df["Date"] >= event_day - pd.Timedelta(days=7)) & (country_df["Date"] < event_day)]
                post_window = country_df[(country_df["Date"] > event_day) & (country_df["Date"] <= event_day + pd.Timedelta(days=7))]

                if not pre_window.empty and not post_window.empty:
                    vix_change = post_window["VIX"].mean() - pre_window["VIX"].mean()
                    index_change = post_window["Index"].mean() - pre_window["Index"].mean()
                    sentiment = classify_sentiment(vix_change, index_change)

                    pre_post_summary.append({
                        "Event": label,
                        "VIX Δ": round(vix_change, 2),
                        "Index Δ": round(index_change, 2),
                        "Investor Mood": sentiment
                    })

            df_sentiment = pd.DataFrame(pre_post_summary)
            st.dataframe(df_sentiment, use_container_width=True)

        with row2_col2:
            mood_map = pd.DataFrame({
                "VIX Change": ["↑", "↓", "↑", "↓", "Small/Neutral"],
                "Index Change": ["↓", "↑", "↑", "↓", "Small/Neutral"],
                "Investor Mood": ["😨 Panic", "📈 Optimistic", "😐 Mixed Reaction", "📉 Uncertain", "😐 Stable Mood"],
                "Interpretation": [
                    "Fear due to volatility spike & market drop",
                    "Confidence returning",
                    "Uncertain: both fear and buying activity",
                    "Market decline but volatility drop suggests stability ahead",
                    "Market unaffected or already priced in"
                ]
            })
            st.markdown("#### 🧠 Investor Mood Interpretation")
            st.dataframe(mood_map, use_container_width=True)



#Tab 3 code (Behavioral insights part)

with tab3:
    st.subheader("🧠 Behavioral Bias Detector – Overreaction vs Overconfidence")

    uploaded_behavior_file = st.file_uploader("📂 Upload VIX + Index CSV (2021–2024)", type="csv", key="behavior_csv")

    if uploaded_behavior_file:
        df_behavior = pd.read_csv(uploaded_behavior_file, parse_dates=["Date"])
        df_behavior["Country"] = df_behavior["Country"].str.title()

        # Select country
        selected_country = st.selectbox("🌍 Select Country", df_behavior["Country"].unique(), key="behavior_country")

        # Filter country data
        country_df = df_behavior[df_behavior["Country"] == selected_country].copy()
        country_df = country_df.sort_values("Date")
        country_df["Index"] = country_df["Index"].interpolate(method="linear")

        # --- Fixed Date Slider using datetime.date ---
        min_date = pd.to_datetime("2021-01-01").date()
        max_date = pd.to_datetime("2024-12-31").date()
        start_date, end_date = st.slider(
            "📅 Select Date Range (Behavioral View)",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key="behavior_date_slider"
        )

        # Convert back to Timestamps
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter data by selected date range
        selected_df = country_df[(country_df["Date"] >= start_date) & (country_df["Date"] <= end_date)]

        if not selected_df.empty:
            # Compute thresholds using selected date range
            mean_vix = selected_df["VIX"].mean()
            std_vix = selected_df["VIX"].std()
            upper_threshold = mean_vix + 2 * std_vix
            lower_threshold = mean_vix - 2 * std_vix
            latest_vix = selected_df["VIX"].iloc[-1]
        
            # Display behavioral signal based on thresholds
            if latest_vix > upper_threshold:
                signal = "🎈 Overreaction Detected"
                animation = "balloons"
            elif latest_vix < lower_threshold:
                signal = "❄️ Overconfidence Detected"
                animation = "snow"
            else:
                signal = "😐 No Strong Behavioral Signal"
                animation = None
        
            st.markdown(f"### 🧠 Behavioral Signal: {signal}")
        
            # Trigger visual feedback
            if animation == "balloons":
                st.balloons()
            elif animation == "snow":
                st.snow()

            # Display threshold context
            st.markdown(f"""
            **Latest VIX:** `{latest_vix:.2f}`  
            **Mean VIX:** `{mean_vix:.2f}`  
            **Upper Bound (+2σ):** `{upper_threshold:.2f}`  
            **Lower Bound (-2σ):** `{lower_threshold:.2f}`
            """)



            # Chart for context
            st.markdown("#### 📊 VIX vs Index (Behavioral Range)")
            fig_behavior = go.Figure()
            fig_behavior.add_trace(go.Scatter(x=selected_df["Date"], y=selected_df["VIX"], name="VIX", line=dict(color="crimson")))
            fig_behavior.add_trace(go.Scatter(x=selected_df["Date"], y=selected_df["Index"], name="Index", line=dict(color="darkcyan"), yaxis="y2"))
            fig_behavior.update_layout(
                yaxis=dict(title="VIX"),
                yaxis2=dict(title="Index", overlaying="y", side="right"),
                height=450,
                margin=dict(t=40, b=30)
            )
            st.plotly_chart(fig_behavior, use_container_width=True)

            st.markdown(
                f"<div class='note-box'>📌 <b>Insight:</b> Based on recent movement, <b>{signal}</b> in {selected_country}. This behavior is detected from the change in VIX vs Index in the selected time range.</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("No data available in the selected range. Try a different date range.")

        if not country_df.empty:
            selected_df = country_df[(country_df["Date"] >= start_date) & (country_df["Date"] <= end_date)]

            if not selected_df.empty:
                st.markdown("### 📉 Behavioral Insights Timeline & Suggestions")

                row2_col1, row2_col2, row2_col3 = st.columns(3)

                with row2_col1:
                    st.markdown("#### 🧭 Behavioral Bias Tracker Timeline")
                    df_tracker = selected_df.copy()
                    df_tracker['Index_Rolling_Mean'] = df_tracker['Index'].rolling(7).mean()
                    df_tracker['Signal'] = df_tracker.apply(
                        lambda row: "Overreaction" if row['VIX'] > 25 and row['Index'] < row['Index_Rolling_Mean']
                        else ("Overconfidence" if row['VIX'] < 12 and row['Index'] > row['Index_Rolling_Mean']
                        else "Neutral"), axis=1
                    )
                    color_map = {"Overreaction": "red", "Overconfidence": "skyblue", "Neutral": "gray"}
                    fig_tracker = go.Figure()
                    for signal_type in df_tracker['Signal'].unique():
                        df_signal = df_tracker[df_tracker['Signal'] == signal_type]
                        fig_tracker.add_trace(go.Scatter(
                            x=df_signal["Date"], y=df_signal["VIX"],
                            mode="markers", name=signal_type,
                            marker=dict(color=color_map[signal_type], size=6),
                            showlegend=True
                        ))
                    fig_tracker.update_layout(
                        yaxis_title="VIX", height=350,
                        margin=dict(t=10, b=20)
                    )
                    st.plotly_chart(fig_tracker, use_container_width=True)
                    st.markdown(note_template("Timeline of behavioral phases using recent VIX and Index interactions."), unsafe_allow_html=True)

                with row2_col2:
                    # --- Row 2: Volatility Rolling Window Chart with Filters ---
                    st.markdown("### 🔄 Rolling Volatility Comparison")

                    # Create volatility columns
                    df_behavior[f"7D_Volatility"] = df_behavior.groupby("Country")["VIX"].transform(lambda x: x.rolling(window=7).std())
                    df_behavior[f"30D_Volatility"] = df_behavior.groupby("Country")["VIX"].transform(lambda x: x.rolling(window=30).std())
                    df_behavior[f"90D_Volatility"] = df_behavior.groupby("Country")["VIX"].transform(lambda x: x.rolling(window=90).std())
                
                    # Filter for selected country again
                    df_vol = df_behavior[df_behavior["Country"] == selected_country].copy()
                
                    # Show toggle multiselect
                    selected_windows = st.multiselect("📈 Select Volatility Window(s)", ["7D", "30D", "90D"], default=["30D"])
                
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(x=df_vol["Date"], y=df_vol["VIX"], name="VIX", line=dict(color="teal", width=2)))
                
                    # Add selected rolling windows
                    if "7D" in selected_windows:
                        fig_vol.add_trace(go.Scatter(x=df_vol["Date"], y=df_vol["7D_Volatility"], name="7D Volatility", line=dict(dash="dot", color="orange")))
                    if "30D" in selected_windows:
                        fig_vol.add_trace(go.Scatter(x=df_vol["Date"], y=df_vol["30D_Volatility"], name="30D Volatility", line=dict(dash="dot", color="blue")))
                    if "90D" in selected_windows:
                        fig_vol.add_trace(go.Scatter(x=df_vol["Date"], y=df_vol["90D_Volatility"], name="90D Volatility", line=dict(dash="dot", color="green")))
                
                    fig_vol.update_layout(
                        height=500,
                        xaxis_title="Date",
                        yaxis_title="Volatility",
                        margin=dict(t=30, b=30),
                        legend=dict(orientation="h")
                    )

                    st.plotly_chart(fig_vol, use_container_width=True)
                    st.markdown(note_template("This chart compares short-term and long-term volatility signals to VIX to help detect market shifts and trading opportunities."), unsafe_allow_html=True)
                

                with row2_col3:
                    st.markdown("### 🧭 Investment Action Advisor")
                
                    # Ensure data exists
                    if not selected_df.empty:
                        latest_vix = selected_df["VIX"].iloc[-1]
                        mean_vix = selected_df["VIX"].mean()
                        std_vix = selected_df["VIX"].std()
                        upper_threshold = mean_vix + 2 * std_vix
                        lower_threshold = mean_vix - 2 * std_vix
                    
                        # Advisor Decision Based on Thresholds
                        if latest_vix > upper_threshold:
                            st.success("🟢 VIX above +2σ → **Overreaction** likely.\n\n**Recommendation:** Market may be undervalued. Consider **BUYING** opportunities.")
                        elif latest_vix < lower_threshold:
                            st.error("🔴 VIX below -2σ → **Overconfidence** likely.\n\n**Recommendation:** Market may be overheated. Consider **SELLING** or applying caution.")
                        else:
                            st.info("⚪ VIX within normal range → No strong behavioral bias.\n\n**Recommendation:** **HOLD** or monitor further signals.")
                    
                        st.markdown(note_template("This advisor uses statistical thresholds to recommend investment actions based on detected market sentiment."), unsafe_allow_html=True)


