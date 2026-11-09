import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="Google Play Store Analytics", layout="wide")

DATA_FILE = "google_play_store_100_cleaned.csv"


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    # Convert text columns such as "10,000+" or "$1.99" to numbers
    for col in ["Installs", "Price", "Reviews", "Rating"]:
        if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(r"[^0-9.]", "", regex=True),
                errors="coerce",
            )
    # Parse a date column if one exists
    for col in data.columns:
        if "date" in col.lower() or "updated" in col.lower():
            parsed = pd.to_datetime(data[col], errors="coerce")
            if parsed.notna().sum() > 0:
                data[col] = parsed
    return data


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


df = load_data(DATA_FILE)

# Column detection (works even if some columns are missing)
text_types = ["object", "string"]
CAT_COL = "Category" if "Category" in df.columns else df.select_dtypes(include=text_types).columns[0]
NAME_COL = "App" if "App" in df.columns else df.columns[0]
num_cols = df.select_dtypes("number").columns.tolist()
date_cols = df.select_dtypes("datetime").columns.tolist()
cat_cols = [
    c for c in df.select_dtypes(include=text_types).columns
    if c != NAME_COL and df[c].nunique() <= 30
]
key_cols = [c for c in ["Installs", "Reviews", "Rating", "Price"] if c in num_cols] or num_cols[:3]

# Simple, consistent chart style
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "axes.titleweight": "bold", "legend.frameon": False,
})
CHART_COLOR = "#4C78A8"

# ------------------------------------------------------------------
# Sidebar: navigation and filters
# ------------------------------------------------------------------
st.sidebar.title("Navigation")
PAGES = ["Dataset Overview", "Data Analysis", "Visualizations", "Insights"]
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.header("Filters")

selected_cats = st.sidebar.multiselect(
    CAT_COL,
    options=sorted(df[CAT_COL].dropna().unique()),
    default=sorted(df[CAT_COL].dropna().unique()),
)
filtered = df[df[CAT_COL].isin(selected_cats)]

extra_cat = [c for c in cat_cols if c != CAT_COL]
if extra_cat:
    pick_col = "Type" if "Type" in extra_cat else extra_cat[0]
    options = ["All"] + sorted(df[pick_col].dropna().unique().tolist())
    choice = st.sidebar.selectbox(pick_col, options)
    if choice != "All":
        filtered = filtered[filtered[pick_col] == choice]

for col in [c for c in ["Rating", "Installs"] if c in num_cols]:
    lo, hi = float(df[col].min()), float(df[col].max())
    if lo < hi:
        rng = st.sidebar.slider(f"{col} range", lo, hi, (lo, hi))
        filtered = filtered[filtered[col].between(rng[0], rng[1])]

top_n = st.sidebar.slider("Top N categories", 3, 20, 10)

st.sidebar.divider()
st.sidebar.write(f"Records selected: **{len(filtered)}** of {len(df)}")
st.sidebar.download_button(
    "Download filtered data (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    "filtered_play_store_data.csv",
    "text/csv",
)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("Google Play Store Analytics")
st.write("Explore app categories, ratings, installations and user engagement.")

if filtered.empty:
    st.warning("No records match the selected filters. Please adjust the sidebar filters.")
    st.stop()

# ================= DATASET OVERVIEW =================
if page == "Dataset Overview":
    with st.container():
        st.subheader("Key Performance Indicators")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", len(filtered))
        c2.metric("Total Categories", filtered[CAT_COL].nunique())
        c3.metric("Total Columns", len(filtered.columns))
        if "Rating" in filtered.columns:
            c4.metric("Average Rating", f"{filtered['Rating'].mean():.2f}")

        st.subheader("Sum and Average of Numerical Attributes")
        cols = st.columns(len(key_cols))
        for col_box, name in zip(cols, key_cols):
            col_box.metric(f"Average {name}", f"{filtered[name].mean():,.2f}")
            col_box.metric(f"Total {name}", f"{filtered[name].sum():,.2f}")

    with st.container():
        st.subheader("Application Records")
        st.dataframe(filtered, width="stretch")

    with st.expander("Dataset Information"):
        a, b = st.columns(2)
        with a:
            st.write("Dataset Shape:", filtered.shape)
            st.write("Column Names:")
            st.write(list(filtered.columns))
            st.write("Data Types:")
            st.write(filtered.dtypes.astype(str))
        with b:
            st.write("Missing Values:")
            st.write(filtered.isnull().sum())

    with st.expander("Statistical Summary"):
        st.dataframe(filtered[num_cols].describe(), width="stretch")

# ================= DATA ANALYSIS =================
elif page == "Data Analysis":
    st.subheader("Data Analysis")
    at1, at2, at3 = st.tabs(["Group-By Analysis", "Correlation", "Top 10 Apps"])

    with at1:
        left, right = st.columns(2)
        with left:
            st.markdown("**Record count per category**")
            counts = (
                filtered.groupby(CAT_COL).size().reset_index(name="Count")
                .sort_values("Count", ascending=False)
            )
            st.dataframe(counts, width="stretch", hide_index=True)
        with right:
            agg_col = st.selectbox("Numeric attribute", num_cols, key="agg")
            agg_fn = st.selectbox("Aggregation", ["mean", "sum", "median", "max", "min"])
            st.markdown(f"**{agg_fn.title()} of {agg_col} per category**")
            agg = (
                filtered.groupby(CAT_COL)[agg_col].agg(agg_fn).reset_index()
                .sort_values(agg_col, ascending=False)
            )
            st.dataframe(agg, width="stretch", hide_index=True)

    with at2:
        if len(num_cols) > 1:
            corr = filtered[num_cols].corr()
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.grid(False)
            im = ax.imshow(corr, cmap="Blues", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label="Correlation coefficient")
            ax.set_title("Correlation Between Numerical Attributes")
            fig.tight_layout()
            show_fig(fig)
        else:
            st.info("At least two numeric columns are needed for a correlation matrix.")

    with at3:
        rank_col = st.selectbox("Rank by", num_cols, key="rank")
        st.dataframe(
            filtered.nlargest(10, rank_col)[[NAME_COL, CAT_COL, rank_col]],
            width="stretch", hide_index=True,
        )

# ================= VISUALIZATIONS =================
elif page == "Visualizations":
    st.subheader("Visualizations")
    vt1, vt2 = st.tabs(["Bar and Line Charts", "Scatter Plot"])

    with vt1:
        col_a, col_b = st.columns(2)

        # ---------- Bar chart ----------
        with col_a:
            st.markdown("**Bar Chart**")
            bar_metric = st.selectbox("Y-axis metric", ["Record Count"] + num_cols, key="bar")
            if bar_metric == "Record Count":
                bar_df = filtered[CAT_COL].value_counts().head(top_n)
                y_label = "Number of Apps"
            else:
                bar_df = (
                    filtered.groupby(CAT_COL)[bar_metric].mean()
                    .sort_values(ascending=False).head(top_n)
                )
                y_label = f"Average {bar_metric}"

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(bar_df.index.astype(str), bar_df.values, color=CHART_COLOR, label=y_label)
            ax.set_title(f"{y_label} by {CAT_COL} (Top {top_n})")
            ax.set_xlabel(CAT_COL)
            ax.set_ylabel(y_label)
            ax.grid(axis="x", visible=False)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            ax.legend()
            fig.tight_layout()
            show_fig(fig)
            st.caption("Taller bars show categories with more apps or higher averages.")

        # ---------- Line chart ----------
        with col_b:
            st.markdown("**Line Chart**")
            line_metric = st.selectbox("Metric", key_cols if key_cols else num_cols, key="line")
            fig, ax = plt.subplots(figsize=(6, 4))
            if date_cols:
                dcol = date_cols[0]
                tmp = filtered.dropna(subset=[dcol])
                line_df = tmp.groupby(tmp[dcol].dt.to_period("M"))[line_metric].mean()
                xs, ys = line_df.index.to_timestamp(), line_df.values
                ax.set_title(f"Average {line_metric} Over Time")
                ax.set_xlabel(dcol)
                note = "Shows how the average changes over time."
            else:
                line_df = (
                    filtered.groupby(CAT_COL)[line_metric].mean()
                    .sort_values(ascending=False).head(top_n)
                )
                xs, ys = line_df.index.astype(str), line_df.values
                ax.set_title(f"Average {line_metric} Across Categories")
                ax.set_xlabel(CAT_COL)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                note = "Shows how the average varies across categories."
            ax.plot(xs, ys, marker="o", color=CHART_COLOR, label=f"Average {line_metric}")
            ax.set_ylabel(f"Average {line_metric}")
            ax.legend()
            fig.tight_layout()
            show_fig(fig)
            st.caption(note)

    with vt2:
        s1, s2, s3 = st.columns(3)
        x_var = s1.selectbox("X-axis", num_cols, index=0, key="sx")
        y_var = s2.selectbox("Y-axis", num_cols, index=min(1, len(num_cols) - 1), key="sy")
        color_by = s3.selectbox("Colour by", [CAT_COL] + [c for c in cat_cols if c != CAT_COL])

        sc_df = filtered[[x_var, y_var, color_by]].dropna().copy()
        top_groups = sc_df[color_by].value_counts().head(8).index
        sc_df[color_by] = sc_df[color_by].where(sc_df[color_by].isin(top_groups), "Other")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        for group, g in sc_df.groupby(color_by):
            ax.scatter(g[x_var], g[y_var], label=str(group), alpha=0.8, s=40)
        ax.set_title(f"{y_var} vs {x_var}")
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.legend(title=color_by, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        show_fig(fig)
        st.caption(
            "Look for upward or downward patterns (relationship), "
            "groups of points (clusters) and isolated points (outliers)."
        )

# ================= INSIGHTS =================
elif page == "Insights":
    st.subheader("Key Insights (based on current filters)")

    top_cat = filtered[CAT_COL].value_counts()
    st.write(
        f"- **Most common category:** {top_cat.index[0]} with {top_cat.iloc[0]} apps "
        f"({top_cat.iloc[0] / len(filtered):.1%} of filtered records)."
    )

    if "Rating" in filtered.columns:
        best = filtered.groupby(CAT_COL)["Rating"].mean().sort_values(ascending=False)
        st.write(
            f"- **Highest average rating:** {best.index[0]} ({best.iloc[0]:.2f}); "
            f"lowest: {best.index[-1]} ({best.iloc[-1]:.2f})."
        )

    if "Installs" in filtered.columns:
        top_app = filtered.loc[filtered["Installs"].idxmax()]
        st.write(
            f"- **Most installed app:** {top_app[NAME_COL]} "
            f"({top_app['Installs']:,.0f} installs, category: {top_app[CAT_COL]})."
        )

    if len(num_cols) > 1:
        pairs = filtered[num_cols].corr().abs().stack()
        pairs = pairs[
            pairs.index.get_level_values(0) != pairs.index.get_level_values(1)
        ].sort_values(ascending=False)
        if not pairs.empty:
            (v1, v2), val = pairs.index[0], pairs.iloc[0]
            st.write(f"- **Strongest numeric relationship:** {v1} and {v2} (|r| = {val:.2f}).")

    with st.expander("Additional observations"):
        st.markdown(
            """
- Categories with more apps do not always have higher ratings.
- Highly installed apps tend to have more reviews, showing user engagement.
- Outliers in the scatter plot highlight exceptionally popular or poorly rated apps.
- Use the sidebar filters to compare Free vs Paid apps or specific categories.
            """
        )