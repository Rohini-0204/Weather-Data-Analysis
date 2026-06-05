import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import folium
from streamlit_folium import st_folium

# ========= DATASET STRUCTURE TRACKER (ADD ONLY) =========
def has_dataset_changed(df):
    signature = tuple(df.columns)
    if "last_signature" not in st.session_state:
        st.session_state.last_signature = signature
        return False
    if st.session_state.last_signature != signature:
        st.session_state.last_signature = signature
        return True
    return False

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Climate Analytics Ultimate",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- COORDINATE MAPPING (Added for Map) ---
COORD_MAP = {
    # Cities
    'Delhi': [28.6139, 77.2090],
    'Mumbai': [19.0760, 72.8777],
    'Kolkata': [22.5726, 88.3639],
    'Chennai': [13.0827, 80.2707],
    'Bangalore': [12.9716, 77.5946],
    'Hyderabad': [17.3850, 78.4867],
    'Ahmedabad': [23.0225, 72.5714],
    'Pune': [18.5204, 73.8567],
    'Jaipur': [26.9124, 75.7873],
    'Lucknow': [26.8467, 80.9462],
    
    # States (Mapped to major city/capital in data)
    'Maharashtra': [19.0760, 72.8777],
    'West Bengal': [22.5726, 88.3639],
    'Tamil Nadu': [13.0827, 80.2707],
    'Karnataka': [12.9716, 77.5946],
    'Telangana': [17.3850, 78.4867],
    'Gujarat': [23.0225, 72.5714],
    'Rajasthan': [26.9124, 75.7873],
    'Uttar Pradesh': [26.8467, 80.9462]
}

# --- 2. DATA LOADING & PROCESSING ---
@st.cache_data
def process_data(df):
    """
    Process the raw dataframe: Date parsing, Feature Engineering, and Mapping.
    """
    try:
        # Date Parsing
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            # We should not reach here if the check is done before calling this function
            st.error("CSV must contain a 'date' column.")
            return pd.DataFrame()
        
        # Feature Engineering
        df['Year'] = df['date'].dt.year
        df['Month'] = df['date'].dt.month
        df['DayOfYear'] = df['date'].dt.dayofyear
        df['Month_Name'] = df['date'].dt.strftime('%b')
        
        # State Mapping (Only applies if 'city' column exists)
        if 'city' in df.columns:
            state_map = {
                'Delhi': 'Delhi', 'Mumbai': 'Maharashtra', 'Pune': 'Maharashtra',
                'Kolkata': 'West Bengal', 'Chennai': 'Tamil Nadu', 'Bangalore': 'Karnataka',
                'Hyderabad': 'Telangana', 'Ahmedabad': 'Gujarat', 'Jaipur': 'Rajasthan',
                'Lucknow': 'Uttar Pradesh'
            }
            # Use map but fill NaN (for cities not in map) with the city name itself or 'Unknown'
            df['State'] = df['city'].map(state_map).fillna(df['city'])
        else:
            # Create a dummy state column if city is missing, to prevent crash
            df['State'] = "Unknown"
            df['city'] = "Unknown"
            
        return df
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()

# ========= AUTO DIAGRAM GENERATOR (UPDATED: COUNTRY COMPARISON) =========
def generate_auto_diagrams(df):
    st.title("📊 Auto-Generated Analytics")
    st.caption("New dataset detected — visualizing patterns automatically")

    # 1. Identify Data Types
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    if not num_cols:
        st.warning("No numeric columns found to visualize.")
        return

    # --- AUTOMATIC SELECTION (HIDDEN) ---
    selected_x = num_cols[0]
    selected_y = num_cols[1] if len(num_cols) > 1 else num_cols[0]

    # --- COUNTRY LOGIC ---
    df_plot = df.copy()
    group_col = None # Used for coloring (hue)
    compare_mode = False
    
    # Check specifically for a column named 'country' (case insensitive)
    country_col_name = next((col for col in df.columns if col.lower() == 'country'), None)
    
    if country_col_name:
        st.markdown("### 🏳️ Country Analysis")
        
        # Get unique countries
        unique_countries = sorted(df[country_col_name].dropna().astype(str).unique())
        
        # Toggle for Comparison
        compare_mode = st.checkbox("Compare Two Countries", value=False)
        
        if compare_mode:
            # --- COMPARISON MODE ---
            c1, c2 = st.columns(2)
            with c1:
                country1 = st.selectbox("Select Country A", unique_countries, index=0)
            with c2:
                # Try to pick a different 2nd country by default
                idx2 = 1 if len(unique_countries) > 1 else 0
                country2 = st.selectbox("Select Country B", unique_countries, index=idx2)
            
            # Filter Data for BOTH countries
            df_plot = df[df[country_col_name].isin([country1, country2])]
            
            # Set Group Column to Country so charts split by Country
            group_col = country_col_name
            st.info(f"Comparing **{country1}** vs **{country2}**.")
            
        else:
            # --- SINGLE COUNTRY MODE ---
            selected_country = st.selectbox("Select a Country to Analyze", unique_countries)
            
            # Filter Data for ONE country
            df_plot = df[df[country_col_name] == selected_country]
            
            # Try to find a sub-region (State/Province) to color code
            state_candidates = [c for c in cat_cols if c.lower() in ['state', 'province', 'territory', 'region', 'admin_name', 'city', 'city_name']]
            other_candidates = [c for c in cat_cols if c != country_col_name and df_plot[c].nunique() > 1 and df_plot[c].nunique() < 50]
            
            if state_candidates:
                group_col = state_candidates[0]
                st.info(f"Analyzing **{selected_country}**. Colored by **{group_col}**.")
            elif other_candidates:
                group_col = other_candidates[0]
                st.info(f"Analyzing **{selected_country}**. Colored by **{group_col}**.")
            else:
                st.info(f"Analyzing **{selected_country}**.")

    # --- GEOSPATIAL MAP ---
    lat_col = next((col for col in df.columns if col.lower() in ['latitude', 'lat']), None)
    lon_col = next((col for col in df.columns if col.lower() in ['longitude', 'lon', 'long']), None)
    
    if lat_col and lon_col and not df_plot.empty:
        st.divider()
        st.subheader(f"📍 Map View")
        
        # Center Map
        center_lat = df_plot[lat_col].mean()
        center_lon = df_plot[lon_col].mean()
        # Zoom out slightly if comparing two countries
        zoom = 4 if compare_mode else 5
        m_auto = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles='CartoDB positron')
        
        # Color Map Logic
        color_map = {}
        if group_col:
            unique_groups = df_plot[group_col].unique()
            # Palette
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', 'black']
            for i, val in enumerate(unique_groups):
                color_map[val] = colors[i % len(colors)]

        # Label Column
        label_col = next((col for col in df.columns if 'name' in col.lower() or 'city' in col.lower() or 'station' in col.lower()), None)

        # Plot points (Limit to 1000)
        for idx, row in df_plot.head(1000).iterrows():
            if pd.notnull(row[lat_col]) and pd.notnull(row[lon_col]):
                point_color = 'blue'
                tooltip_text = f"Row {idx}"
                
                if group_col:
                    cat_val = row[group_col]
                    point_color = color_map.get(cat_val, 'blue')
                    tooltip_text = f"{group_col}: {cat_val}"
                    if label_col: tooltip_text = f"{row[label_col]} ({cat_val})"
                elif label_col:
                    tooltip_text = f"{row[label_col]}"

                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=5,
                    color=point_color,
                    fill=True,
                    fill_color=point_color,
                    fill_opacity=0.7,
                    tooltip=tooltip_text
                ).add_to(m_auto)
        
        st_folium(m_auto, height=400, use_container_width=True)
    
    # --- ROW 1: DISTRIBUTION & SPREAD ---
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"1. Distribution of {selected_x}")
        fig1, ax1 = plt.subplots(figsize=(8, 5)) 
        
        # If comparing, 'hue' is Country. If single, 'hue' is State (if available).
        if group_col and df_plot[group_col].nunique() <= 15:
            sns.histplot(data=df_plot, x=selected_x, hue=group_col, kde=True, element="step", ax=ax1)
        else:
            sns.histplot(df_plot[selected_x].dropna(), kde=True, color="#4c72b0", ax=ax1)
            
        ax1.set_title(f"Histogram: {selected_x}")
        ax1.grid(True, alpha=0.2)
        st.pyplot(fig1)

    with col2:
        st.subheader(f"2. Value Spread")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        if group_col and df_plot[group_col].nunique() <= 10:
            sns.violinplot(data=df_plot, x=selected_x, y=group_col, palette="muted", ax=ax2)
        else:
            sns.violinplot(x=df_plot[selected_x], color="#dd8452", inner="quartile", ax=ax2)
            
        ax2.set_title(f"Spread: {selected_x}")
        ax2.grid(True, axis='x', alpha=0.2)
        st.pyplot(fig2)

    # --- ROW 2: RELATIONSHIPS ---
    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader(f"3. Relationship: {selected_x} vs {selected_y}")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        
        if len(num_cols) > 1:
            if group_col and df_plot[group_col].nunique() <= 15:
                sns.scatterplot(data=df_plot, x=selected_x, y=selected_y, hue=group_col, alpha=0.7, s=60, ax=ax3)
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            else:
                sns.regplot(data=df_plot, x=selected_x, y=selected_y, scatter_kws={'alpha':0.5, 's':30}, line_kws={'color':'red'}, ax=ax3)
            ax3.set_title(f"Correlation: {selected_x} vs {selected_y}")
        else:
            ax3.text(0.5, 0.5, "Need >1 numeric column", ha='center')
        st.pyplot(fig3)

    with col4:
        st.subheader("4. Heatmap Correlation")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        if len(num_cols) > 1:
            cols_to_corr = num_cols[:10] 
            sns.heatmap(df_plot[cols_to_corr].corr(), annot=True, fmt=".1f", cmap="coolwarm", ax=ax4)
        else:
            ax4.text(0.5, 0.5, "Not enough data", ha='center')
        st.pyplot(fig4)

    # --- ROW 3: TRENDS (Only if Date exists) ---
    if 'date' in df.columns:
        st.divider()
        st.subheader(f"5. Time Series Trend: {selected_x}")
        fig5, ax5 = plt.subplots(figsize=(12, 4))
        
        df_sorted = df_plot.sort_values('date')
        
        if group_col and df_plot[group_col].nunique() <= 5:
            sns.lineplot(data=df_sorted, x='date', y=selected_x, hue=group_col, ax=ax5)
        else:
            ax5.plot(pd.to_datetime(df_sorted['date']), df_sorted[selected_x], color='#2ca02c', linewidth=1.5)
        
        ax5.set_title(f"{selected_x} over Time")
        ax5.set_ylabel(selected_x)
        ax5.grid(True, alpha=0.3)
        st.pyplot(fig5)

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🎛️ Command Center")

# --- FILE UPLOADER SECTION (MODIFIED) ---
uploaded_file = st.sidebar.file_uploader("📂 Upload Weather Data (CSV)", type=['csv'])

if uploaded_file is not None:
    # Load from User Upload
    df_raw_input = pd.read_csv(uploaded_file)
    
    # ========= CHECK FOR AUTO MODE *BEFORE* PROCESSING =========
    if 'date' not in df_raw_input.columns:
        # If 'date' is missing, run Auto Mode immediately and STOP the rest of the app
        generate_auto_diagrams(df_raw_input)
        st.stop()

    # If 'date' exists, proceed to normal processing
    df_raw = process_data(df_raw_input)
    
    # ========= RESET IF NEW DATASET STRUCTURE =========
    if has_dataset_changed(df_raw_input):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

else:
    # Load from Default Local File
    try:
        df_raw_input = pd.read_csv('india_2000_2024_daily_weather.csv')
        # Check default file for date too, just in case
        if 'date' in df_raw_input.columns:
            df_raw = process_data(df_raw_input)
        else:
             df_raw = pd.DataFrame()
    except FileNotFoundError:
        df_raw = pd.DataFrame()

# --- MAIN LOGIC ---
if not df_raw.empty:
    
    # 1. Level Selection
    if 'city' in df_raw.columns and 'State' in df_raw.columns:
        analysis_level = st.sidebar.radio("Analysis Level", ["City", "State"], horizontal=True)
        
        if analysis_level == "State":
            # Group by State
            group_cols = ['State', 'date', 'Year', 'Month', 'DayOfYear', 'Month_Name']
            df = df_raw.groupby(group_cols).mean(numeric_only=True).reset_index()
            entity_col = 'State'
            entity_list = sorted(df['State'].dropna().unique())
        else:
            df = df_raw.copy()
            entity_col = 'city'
            entity_list = sorted(df['city'].unique())
    else:
        df = df_raw.copy()
        analysis_level = "Custom"
        st.sidebar.warning("Standard columns (city, State) not found. Functionality may be limited.")
        entity_col = df.columns[0] if not df.columns.empty else 'Unknown'
        entity_list = sorted(df[entity_col].unique()) if entity_col != 'Unknown' else []

    if entity_list:
        # 2. Primary Selection
        selected_primary = st.sidebar.selectbox(f"Select {analysis_level}", entity_list)

        # 3. Comparison Toggle
        st.sidebar.divider()
        enable_compare = st.sidebar.toggle(" Enable Comparison Mode", value=False)
        
        if enable_compare:
            mode = "Compare Two"
            secondary_list = [x for x in entity_list if x != selected_primary]
            selected_secondary = st.sidebar.selectbox(f"Select Comparison {analysis_level}", secondary_list)
            
            # Filter Data
            df_filtered = df[df[entity_col].isin([selected_primary, selected_secondary])]
            
            # Colors: Primary = Red, Secondary = Orange
            color_primary = '#ff4b4b' 
            color_secondary = '#FFA500' 
        else:
            mode = "Single Analysis"
            selected_secondary = None
            df_filtered = df[df[entity_col] == selected_primary]
            color_primary = '#ff4b4b' # Red

        # 4. Time Filter
        st.sidebar.divider()
        min_year, max_year = int(df['Year'].min()), int(df['Year'].max())
        year_range = st.sidebar.slider("📅 Analysis Period", min_year, max_year, (min_year, max_year))
        df_final = df_filtered[(df_filtered['Year'] >= year_range[0]) & (df_filtered['Year'] <= year_range[1])]

        # --- HEADER ---
        if mode == "Single Analysis":
            st.title(f"📊 {selected_primary} Analytics")
        else:
            st.title(f" {selected_primary} vs {selected_secondary}")
        st.caption(f"Showing all 8 Advanced Diagrams | Period: {year_range[0]}-{year_range[1]}")

        # --- NEW SECTION: GEOSPATIAL MAP ---
        with st.container():
            
            # Initialize Map centered on India
            m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='CartoDB positron')

            # Helper to add marker
            def add_marker(name, role, color):
                if name in COORD_MAP:
                    coords = COORD_MAP[name]
                    folium.Marker(
                        location=coords,
                        popup=f"<b>{name}</b> ({role})",
                        tooltip=f"{role}: {name}",
                        icon=folium.Icon(color=color, icon='info-sign')
                    ).add_to(m)

            # Add Primary Marker
            add_marker(selected_primary, "Primary", "red")

            # Add Secondary Marker if enabled
            if mode == "Compare Two" and selected_secondary:
                add_marker(selected_secondary, "Secondary", "orange")

            # Display Map
            st_folium(m, height=300, use_container_width=True)
            st.divider()

        # --- 4. VISUALIZATION TABS ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧩 Patterns", "🌊 Distributions", "🎻 Relationships", "🚨 Anomalies", "📈 Trends"])

        # === TAB 1: PATTERNS (Correlation Heatmap & Polar) ===
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("1. Variable Correlation Matrix")
                
                # Helper to prep data for correlation (numeric only, drop time columns)
                def prep_corr_data(d):
                    d_num = d.select_dtypes(include=np.number)
                    cols_to_drop = ['Year', 'Month', 'DayOfYear']
                    return d_num.drop(columns=[c for c in cols_to_drop if c in d_num.columns], errors='ignore')

                if mode == "Compare Two":
                    # Compare Logic: Show Primary
                    st.markdown(f"**{selected_primary}**")
                    data_p1 = prep_corr_data(df_final[df_final[entity_col] == selected_primary])
                    
                    if not data_p1.empty:
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        sns.heatmap(data_p1.corr(), annot=True, fmt=".2f", cmap='coolwarm', 
                                    cbar=True, cbar_kws={'label': 'Correlation'}, 
                                    ax=ax1, annot_kws={"size": 8})
                        st.pyplot(fig1)
                    
                    # Show Secondary
                    st.markdown(f"**{selected_secondary}**")
                    data_p2 = prep_corr_data(df_final[df_final[entity_col] == selected_secondary])
                    
                    if not data_p2.empty:
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        sns.heatmap(data_p2.corr(), annot=True, fmt=".2f", cmap='coolwarm', 
                                    cbar=True, cbar_kws={'label': 'Correlation'}, 
                                    ax=ax2, annot_kws={"size": 8})
                        st.pyplot(fig2)

                else:
                    # Single Mode
                    data_corr = prep_corr_data(df_final)
                    
                    if not data_corr.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.heatmap(data_corr.corr(), annot=True, fmt=".2f", cmap='coolwarm', 
                                    cbar=True, cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("Not enough numeric data for correlation.")

            with col2:
                st.subheader("2. Seasonal Cycle (Polar Plot)")
                agg = df_final.groupby([entity_col, 'Month'])['temperature_2m_max'].mean().reset_index()
                angles = np.linspace(0, 2*np.pi, 12, endpoint=False).tolist()
                angles += angles[:1]
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig_p, ax_p = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                
                # Primary Line
                v1 = agg[agg[entity_col] == selected_primary]['temperature_2m_max'].tolist()
                if len(v1) == 12: 
                    v1 += v1[:1]
                    ax_p.plot(angles, v1, color=color_primary, linewidth=2, label=selected_primary)
                    ax_p.fill(angles, v1, color=color_primary, alpha=0.1)
                
                # Secondary Line (if compare)
                if mode == "Compare Two":
                    v2 = agg[agg[entity_col] == selected_secondary]['temperature_2m_max'].tolist()
                    if len(v2) == 12:
                        v2 += v2[:1]
                        ax_p.plot(angles, v2, color=color_secondary, linewidth=2, label=selected_secondary)
                        ax_p.fill(angles, v2, color=color_secondary, alpha=0.1)
                
                ax_p.set_xticks(angles[:-1])
                ax_p.set_xticklabels(months)
                ax_p.legend(loc='lower right', bbox_to_anchor=(1.3, 0))
                st.pyplot(fig_p)

        # === TAB 2: DISTRIBUTIONS (KDE & Ridge) ===
        with tab2:
            c_dist1, c_dist2 = st.columns(2)
            
            with c_dist1:
                st.subheader("3. Probability Distribution (KDE)")
                fig_k, ax_k = plt.subplots(figsize=(8, 6))
                if mode == "Compare Two":
                    sns.kdeplot(data=df_final, x='temperature_2m_max', hue=entity_col, fill=True, 
                                palette={selected_primary: color_primary, selected_secondary: color_secondary}, ax=ax_k)
                else:
                    sns.kdeplot(data=df_final, x='temperature_2m_max', fill=True, color=color_primary, ax=ax_k)
                st.pyplot(fig_k)
                
            with c_dist2:
                st.subheader("4. Climate Evolution (Ridge Plot)")
                target_ridge = selected_primary # Default
                if mode == "Compare Two":
                    target_ridge = st.selectbox("Select Entity for Ridge Plot", [selected_primary, selected_secondary])
                
                data_ridge = df_final[df_final[entity_col] == target_ridge]
                years_sel = sorted(data_ridge['Year'].unique())[::max(1, len(data_ridge['Year'].unique())//8)]
                
                fig_r, ax_r = plt.subplots(figsize=(8, len(years_sel)*0.5 + 2))
                pal = sns.color_palette("coolwarm", n_colors=len(years_sel))
                
                for i, year in enumerate(years_sel):
                    sub = data_ridge[data_ridge['Year'] == year]
                    if len(sub) > 10:
                        density = stats.gaussian_kde(sub['temperature_2m_max'])
                        xs = np.linspace(sub['temperature_2m_max'].min(), sub['temperature_2m_max'].max(), 200)
                        ys = density(xs)
                        ys = ys/ys.max()
                        ax_r.fill_between(xs, ys + i*0.5, i*0.5, color=pal[i], alpha=0.7)
                        ax_r.text(xs.min(), i*0.5, str(year), fontsize=9, fontweight='bold')
                
                ax_r.set_yticks([])
                ax_r.spines['left'].set_visible(False); ax_r.spines['right'].set_visible(False); ax_r.spines['top'].set_visible(False)
                st.pyplot(fig_r)

        # === TAB 3: RELATIONSHIPS (Hexbin & Violin) ===
        with tab3:
            c_rel1, c_rel2 = st.columns(2)
            
            with c_rel1:
                st.subheader("5. Weather Density (Hexbin)")
                target_hex = selected_primary # Default
                if mode == "Compare Two":
                    target_hex = st.selectbox("Select Entity for Hexbin", [selected_primary, selected_secondary])
                
                d_hex = df_final[df_final[entity_col] == target_hex]
                fig_h, ax_h = plt.subplots(figsize=(8, 6))
                
                y_col = 'wind_speed_10m_max' if 'wind_speed_10m_max' in d_hex.columns else d_hex.select_dtypes(include=np.number).columns[1]
                
                try:
                    hb = ax_h.hexbin(d_hex['temperature_2m_max'], d_hex[y_col], gridsize=25, cmap='Blues', mincnt=1)
                    fig_h.colorbar(hb, ax=ax_h, label='Days Count')
                    ax_h.set_xlabel("Temp (°C)"); ax_h.set_ylabel(y_col)
                    ax_h.set_title(f"Density: {target_hex}")
                    st.pyplot(fig_h)
                except Exception as e:
                    st.info(f"Could not render Hexbin (missing columns): {e}")

            with c_rel2:
                st.subheader("6. Monthly Volatility (Violin)")
                fig_v, ax_v = plt.subplots(figsize=(10, 6))
                sns.violinplot(data=df_final, x='Month_Name', y='temperature_2m_max', hue=entity_col, 
                               split=(mode=="Compare Two"), 
                               palette={selected_primary: color_primary, selected_secondary: color_secondary} if mode=="Compare Two" else "magma",
                               ax=ax_v)
                st.pyplot(fig_v)

        # === TAB 4: ANOMALIES ===
        with tab4:
            st.subheader("7. Statistical Anomaly Detection")
            
            sigma = st.slider("Anomaly Threshold (Sigma)", 2.0, 4.0, 2.5)
            window = 30
            
            def get_anomalies(d):
                d = d.sort_values('date')
                roll_mean = d['temperature_2m_max'].rolling(window).mean()
                roll_std = d['temperature_2m_max'].rolling(window).std()
                high = d[d['temperature_2m_max'] > (roll_mean + sigma * roll_std)]
                low = d[d['temperature_2m_max'] < (roll_mean - sigma * roll_std)]
                return d, roll_mean, high, low

            # Primary Data
            d_a, mean_a, high_a, low_a = get_anomalies(df_final[df_final[entity_col] == selected_primary])
            
            # Plot Time Series
            fig_ts, ax_ts = plt.subplots(figsize=(12, 5))
            ax_ts.plot(d_a['date'], d_a['temperature_2m_max'], color='gray', alpha=0.3, label=f'{selected_primary} Daily')
            ax_ts.plot(d_a['date'], mean_a, color=color_primary, linewidth=1.5, label=f'{selected_primary} Trend')
            ax_ts.scatter(high_a['date'], high_a['temperature_2m_max'], color='red', s=20, zorder=3, label='Heat Anom')
            ax_ts.scatter(low_a['date'], low_a['temperature_2m_max'], color='blue', s=20, zorder=3, label='Cold Anom')
            
            if mode == "Compare Two":
                d_b, mean_b, high_b, low_b = get_anomalies(df_final[df_final[entity_col] == selected_secondary])
                ax_ts.plot(d_b['date'], mean_b, color=color_secondary, linewidth=1.5, linestyle='--', label=f'{selected_secondary} Trend')
            
            ax_ts.legend()
            ax_ts.set_title("Time Series with Anomalies")
            st.pyplot(fig_ts)
            
            # Bar Chart Comparison
            st.divider()
            st.markdown("**Anomaly Counts**")
            h_cnt_a, l_cnt_a = len(high_a), len(low_a)
            
            if mode == "Compare Two":
                h_cnt_b, l_cnt_b = len(high_b), len(low_b)
                df_bar = pd.DataFrame({
                    'Entity': [selected_primary, selected_primary, selected_secondary, selected_secondary],
                    'Type': ['Heat', 'Cold', 'Heat', 'Cold'],
                    'Count': [h_cnt_a, l_cnt_a, h_cnt_b, l_cnt_b]
                })
                fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
                sns.barplot(data=df_bar, x='Type', y='Count', hue='Entity', palette={selected_primary: color_primary, selected_secondary: color_secondary}, ax=ax_bar)
                st.pyplot(fig_bar)
            else:
                c1, c2 = st.columns(2)
                c1.metric("Heat Anomalies", h_cnt_a)
                c2.metric("Cold Anomalies", l_cnt_a)

        # === TAB 5: TRENDS (UPDATED COLORS) ===
        with tab5:
            st.subheader("8. Trend Analysis (Temperature & Rainfall)")
            
            # Identify Rain Column
            rain_cols = [c for c in df_final.columns if 'rain' in c.lower() or 'precip' in c.lower()]
            rain_col = rain_cols[0] if rain_cols else None
            
            if not rain_col:
                st.warning("Rainfall data not found in columns. Only showing Temperature.")
            
            # Controls
            trend_period = st.radio("Select Trend Aggregation", ["Daily (Rolling)", "Monthly Average", "Yearly Average"], horizontal=True)
            
            fig_trend, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx() # Dual Axis
            
            def get_trend_data(df_sub, period, r_col):
                df_sub = df_sub.sort_values('date')
                if period == "Monthly Average":
                    # Temp: Mean, Rain: Sum
                    t = df_sub.set_index('date').resample('M')['temperature_2m_max'].mean()
                    r = df_sub.set_index('date').resample('M')[r_col].sum() if r_col else None
                    return t, r
                elif period == "Yearly Average":
                    t = df_sub.set_index('date').resample('Y')['temperature_2m_max'].mean()
                    r = df_sub.set_index('date').resample('Y')[r_col].sum() if r_col else None
                    return t, r
                else:
                    return df_sub, df_sub # Raw data
            
            # --- Primary Plot ---
            d_p = df_final[df_final[entity_col] == selected_primary]
            
            if trend_period == "Daily (Rolling)":
                win_size = st.slider("Rolling Window Size (Days)", 7, 365, 30)
                d_p = d_p.sort_values('date')
                
                # Temp (Red)
                ma_p = d_p['temperature_2m_max'].rolling(win_size).mean()
                ax1.plot(d_p['date'], ma_p, color=color_primary, linewidth=2, label=f'{selected_primary} Temp')
                
                # Rain (Blue)
                if rain_col:
                    r_ma_p = d_p[rain_col].rolling(win_size).mean()
                    ax2.fill_between(d_p['date'], r_ma_p, color='blue', alpha=0.2, label=f'{selected_primary} Rain')

            else:
                # Monthly/Yearly
                t_p, r_p = get_trend_data(d_p, trend_period, rain_col)
                # Temp (Red)
                ax1.plot(t_p.index, t_p.values, color=color_primary, marker='o', linestyle='-', linewidth=2, label=f'{selected_primary} Temp')
                if r_p is not None:
                    # Rain (Blue)
                    ax2.bar(r_p.index, r_p.values, color='blue', alpha=0.3, width=20 if trend_period=="Monthly Average" else 200, label=f'{selected_primary} Rain')

            # --- Secondary Plot ---
            if mode == "Compare Two":
                d_s = df_final[df_final[entity_col] == selected_secondary]
                
                if trend_period == "Daily (Rolling)":
                    d_s = d_s.sort_values('date')
                    # Temp (Orange - as requested)
                    ma_s = d_s['temperature_2m_max'].rolling(win_size).mean()
                    ax1.plot(d_s['date'], ma_s, color=color_secondary, linewidth=2, linestyle='--', label=f'{selected_secondary} Temp')
                    if rain_col:
                        # Rain (SkyBlue)
                        r_ma_s = d_s[rain_col].rolling(win_size).mean()
                        ax2.fill_between(d_s['date'], r_ma_s, color='skyblue', alpha=0.2, label=f'{selected_secondary} Rain')
                else:
                    t_s, r_s = get_trend_data(d_s, trend_period, rain_col)
                    # Temp (Orange - as requested)
                    ax1.plot(t_s.index, t_s.values, color=color_secondary, marker='s', linestyle='--', linewidth=2, label=f'{selected_secondary} Temp')
                    if r_s is not None:
                         # Rain (SkyBlue)
                        ax2.bar(r_s.index, r_s.values, color='skyblue', alpha=0.3, width=20 if trend_period=="Monthly Average" else 200, label=f'{selected_secondary} Rain')
            
            ax1.set_ylabel("Max Temp (°C)", color='black')
            ax2.set_ylabel(f"Rainfall ({'Sum' if 'Average' in trend_period else 'Mean'} mm)", color='blue')
            ax1.set_title(f"Climograph: Temperature & Rainfall Trends ({trend_period})")
            
            # Combine legends
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
            
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig_trend)

    else:
        st.error("No valid data found. Please upload a CSV with 'city' and 'date' columns.")

else:
    st.info("👋 Welcome! Please upload a CSV file or ensure 'india_2000_2024_daily_weather.csv' is in the directory to start.")