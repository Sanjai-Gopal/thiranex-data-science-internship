import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. GENERATE REALISTIC RAW DATASET
# ──────────────────────────────────────────────
np.random.seed(42)
n = 500

raw_data = {
    'CustomerID': range(1001, 1001 + n),
    'Age': np.random.randint(18, 70, n).astype(float),
    'Gender': np.random.choice(['Male', 'Female', 'male', 'FEMALE', None], n, p=[0.35, 0.35, 0.1, 0.1, 0.1]),
    'Annual_Income_K': np.random.normal(60, 20, n).round(2),
    'Spending_Score': np.random.randint(1, 101, n).astype(float),
    'Region': np.random.choice(['North', 'South', 'East', 'West', 'NORTH', 'south', None], n, p=[0.2,0.2,0.2,0.2,0.05,0.05,0.1]),
    'Purchase_Count': np.random.poisson(8, n).astype(float),
    'Last_Purchase_Days': np.random.exponential(30, n).round(0),
    'Loyalty_Years': np.random.uniform(0, 10, n).round(1),
    'Satisfaction_Rating': np.random.choice([1,2,3,4,5,None,99], n, p=[0.05,0.1,0.2,0.35,0.2,0.05,0.05]),
}

df_raw = pd.DataFrame(raw_data)

# Inject issues
df_raw.loc[np.random.choice(n, 40, replace=False), 'Age'] = np.nan
df_raw.loc[np.random.choice(n, 20, replace=False), 'Annual_Income_K'] = -999
df_raw.loc[np.random.choice(n, 15, replace=False), 'Annual_Income_K'] = 9999
df_raw.loc[np.random.choice(n, 10, replace=False), 'Spending_Score'] = np.nan
df_raw.loc[np.random.choice(n, 30, replace=False), 'Purchase_Count'] = np.nan
df_raw = pd.concat([df_raw, df_raw.iloc[:15]], ignore_index=True)  # duplicates

df_raw.to_csv('/home/claude/data_cleaning_project/raw_customer_data.csv', index=False)
print("Raw dataset created:", df_raw.shape)
print("\nMissing values:\n", df_raw.isnull().sum())
print("\nDuplicate rows:", df_raw.duplicated().sum())

# ──────────────────────────────────────────────
# 2. DATA CLEANING PIPELINE
# ──────────────────────────────────────────────
df = df_raw.copy()
issues_log = []

# Step 1: Remove duplicates
before = len(df)
df = df.drop_duplicates()
removed_dupes = before - len(df)
issues_log.append(f"Removed {removed_dupes} duplicate rows")

# Step 2: Standardize Gender
df['Gender'] = df['Gender'].str.strip().str.capitalize()
df['Gender'] = df['Gender'].replace({'Male': 'Male', 'Female': 'Female'})
df.loc[~df['Gender'].isin(['Male', 'Female']), 'Gender'] = np.nan

# Step 3: Standardize Region
df['Region'] = df['Region'].str.strip().str.capitalize()

# Step 4: Handle outliers in Annual_Income_K
invalid_income = df['Annual_Income_K'].isin([-999, 9999])
df.loc[invalid_income, 'Annual_Income_K'] = np.nan
issues_log.append(f"Replaced {invalid_income.sum()} invalid income values with NaN")

# Step 5: Handle invalid Satisfaction_Rating
df.loc[df['Satisfaction_Rating'] == 99, 'Satisfaction_Rating'] = np.nan

# Step 6: Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Annual_Income_K'].fillna(df['Annual_Income_K'].median(), inplace=True)
df['Spending_Score'].fillna(df['Spending_Score'].median(), inplace=True)
df['Purchase_Count'].fillna(df['Purchase_Count'].median(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Region'].fillna(df['Region'].mode()[0], inplace=True)
df['Satisfaction_Rating'].fillna(df['Satisfaction_Rating'].median(), inplace=True)
issues_log.append(f"Filled missing values using median/mode imputation")

# Step 7: Feature engineering
df['Income_Group'] = pd.cut(df['Annual_Income_K'], bins=[0,30,60,90,200],
                             labels=['Low','Medium','High','Very High'])
df['Age_Group'] = pd.cut(df['Age'], bins=[17,25,35,50,70],
                          labels=['Youth','Young Adult','Middle Age','Senior'])
df['High_Value'] = ((df['Spending_Score'] > 70) & (df['Annual_Income_K'] > 60)).astype(int)

df.to_csv('/home/claude/data_cleaning_project/cleaned_customer_data.csv', index=False)
print("\nCleaned dataset:", df.shape)
print("Remaining missing values:", df.isnull().sum().sum())

# ──────────────────────────────────────────────
# 3. VISUALIZATION DASHBOARD
# ──────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 16), facecolor='#0f1117')
gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38,
              left=0.06, right=0.97, top=0.92, bottom=0.07)

TEAL   = '#00d4aa'
CORAL  = '#ff6b6b'
GOLD   = '#ffd166'
PURPLE = '#a78bfa'
BLUE   = '#60a5fa'
BG     = '#0f1117'
PANEL  = '#1a1d2e'
TEXT   = '#e2e8f0'
MUTED  = '#64748b'

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3748')
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

# Title
fig.text(0.5, 0.96, 'Customer Data Cleaning & Visualization Dashboard',
         ha='center', va='top', fontsize=20, fontweight='bold', color=TEXT,
         fontfamily='monospace')
fig.text(0.5, 0.935, 'Thiranex Data Science Internship  •  Sanjai G  •  Task 1',
         ha='center', va='top', fontsize=11, color=MUTED)

# ── Plot 1: Data Quality Before vs After ──
ax1 = fig.add_subplot(gs[0, 0])
missing_before = df_raw.isnull().sum()
missing_before = missing_before[missing_before > 0]
missing_after = df[missing_before.index].isnull().sum()
x = np.arange(len(missing_before))
w = 0.35
bars1 = ax1.bar(x - w/2, missing_before.values, w, color=CORAL, alpha=0.85, label='Before')
bars2 = ax1.bar(x + w/2, missing_after.values, w, color=TEAL, alpha=0.85, label='After')
ax1.set_xticks(x)
ax1.set_xticklabels(missing_before.index, rotation=35, ha='right', fontsize=7.5)
ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
style_ax(ax1, '① Missing Values: Before vs After')

# ── Plot 2: Age Distribution ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df['Age'], bins=25, color=PURPLE, alpha=0.85, edgecolor='#0f1117', linewidth=0.5)
ax2.axvline(df['Age'].mean(), color=GOLD, linestyle='--', linewidth=1.5, label=f"Mean: {df['Age'].mean():.1f}")
ax2.axvline(df['Age'].median(), color=CORAL, linestyle=':', linewidth=1.5, label=f"Median: {df['Age'].median():.1f}")
ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
style_ax(ax2, '② Age Distribution')

# ── Plot 3: Income vs Spending Score ──
ax3 = fig.add_subplot(gs[0, 2])
scatter = ax3.scatter(df['Annual_Income_K'], df['Spending_Score'],
                      c=df['Satisfaction_Rating'], cmap='plasma',
                      alpha=0.6, s=18, edgecolors='none')
cb = plt.colorbar(scatter, ax=ax3)
cb.ax.tick_params(colors=MUTED, labelsize=8)
cb.set_label('Satisfaction', color=MUTED, fontsize=8)
ax3.set_xlabel('Annual Income (K)', fontsize=9)
ax3.set_ylabel('Spending Score', fontsize=9)
style_ax(ax3, '③ Income vs Spending Score')

# ── Plot 4: Gender Distribution ──
ax4 = fig.add_subplot(gs[0, 3])
gender_counts = df['Gender'].value_counts()
wedges, texts, autotexts = ax4.pie(gender_counts.values,
    labels=gender_counts.index, autopct='%1.1f%%',
    colors=[BLUE, CORAL], startangle=90,
    textprops={'color': TEXT, 'fontsize': 9},
    wedgeprops={'edgecolor': BG, 'linewidth': 2})
for at in autotexts:
    at.set_color(BG); at.set_fontweight('bold')
style_ax(ax4, '④ Gender Distribution')
ax4.set_facecolor(PANEL)

# ── Plot 5: Spending Score by Region ──
ax5 = fig.add_subplot(gs[1, 0:2])
region_order = df.groupby('Region')['Spending_Score'].median().sort_values(ascending=False).index
region_colors = [TEAL, PURPLE, CORAL, GOLD, BLUE]
bp = ax5.boxplot([df[df['Region']==r]['Spending_Score'].values for r in region_order],
                  labels=region_order,
                  patch_artist=True,
                  medianprops={'color': BG, 'linewidth': 2})
for patch, color in zip(bp['boxes'], region_colors):
    patch.set_facecolor(color); patch.set_alpha(0.75)
for element in ['whiskers','caps','fliers']:
    for item in bp[element]:
        item.set_color(MUTED)
style_ax(ax5, '⑤ Spending Score Distribution by Region')
ax5.set_ylabel('Spending Score', fontsize=9)

# ── Plot 6: Avg Satisfaction by Income Group ──
ax6 = fig.add_subplot(gs[1, 2])
sat_income = df.groupby('Income_Group', observed=True)['Satisfaction_Rating'].mean()
bars = ax6.bar(sat_income.index, sat_income.values,
               color=[BLUE, TEAL, GOLD, CORAL], alpha=0.85, edgecolor=BG, linewidth=1.5)
for bar, val in zip(bars, sat_income.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
             f'{val:.2f}', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
ax6.set_ylim(0, 5.5)
ax6.set_ylabel('Avg Satisfaction', fontsize=9)
style_ax(ax6, '⑥ Satisfaction by Income Group')

# ── Plot 7: Heatmap of correlations ──
ax7 = fig.add_subplot(gs[1, 3])
corr_cols = ['Age','Annual_Income_K','Spending_Score','Purchase_Count','Loyalty_Years','Satisfaction_Rating']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=ax7, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, annot_kws={'size': 7},
            linewidths=0.5, linecolor='#0f1117',
            cbar_kws={'shrink': 0.8})
ax7.tick_params(labelsize=7.5, colors=MUTED)
ax7.set_facecolor(PANEL)
ax7.set_title('⑦ Feature Correlation Heatmap', color=TEXT, fontsize=11, fontweight='bold', pad=10)

# ── Plot 8: Purchase Count by Age Group ──
ax8 = fig.add_subplot(gs[2, 0:2])
age_purchase = df.groupby('Age_Group', observed=True)['Purchase_Count'].agg(['mean','std'])
x_pos = np.arange(len(age_purchase))
ax8.bar(x_pos, age_purchase['mean'], yerr=age_purchase['std'],
        color=[GOLD, TEAL, PURPLE, CORAL], alpha=0.85,
        error_kw={'color': TEXT, 'capsize': 5, 'linewidth': 1.5},
        edgecolor=BG, linewidth=1.5)
ax8.set_xticks(x_pos)
ax8.set_xticklabels(age_purchase.index, fontsize=9)
ax8.set_ylabel('Avg Purchase Count', fontsize=9)
style_ax(ax8, '⑧ Average Purchases by Age Group (±1 SD)')

# ── Plot 9: High Value Customer Breakdown ──
ax9 = fig.add_subplot(gs[2, 2])
hv = df.groupby(['Region', 'High_Value'], observed=True).size().unstack(fill_value=0)
hv_pct = hv.div(hv.sum(axis=1), axis=0) * 100
hv_pct.plot(kind='bar', ax=ax9, color=[MUTED, TEAL], edgecolor=BG, linewidth=1)
ax9.set_xticklabels(hv_pct.index, rotation=30, ha='right', fontsize=8.5)
ax9.set_ylabel('Percentage (%)', fontsize=9)
ax9.legend(['Standard', 'High Value'], fontsize=8, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
style_ax(ax9, '⑨ High-Value Customers by Region')

# ── Plot 10: Loyalty vs Spending ──
ax10 = fig.add_subplot(gs[2, 3])
for grp, color in zip(['Low','Medium','High','Very High'], [BLUE, TEAL, GOLD, CORAL]):
    subset = df[df['Income_Group'] == grp]
    ax10.scatter(subset['Loyalty_Years'], subset['Spending_Score'],
                 alpha=0.5, s=14, color=color, label=grp)
# Trend line
z = np.polyfit(df['Loyalty_Years'], df['Spending_Score'], 1)
p = np.poly1d(z)
xline = np.linspace(0, 10, 100)
ax10.plot(xline, p(xline), color=TEXT, linewidth=1.5, linestyle='--', alpha=0.7)
ax10.legend(title='Income', fontsize=7.5, title_fontsize=8,
            facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
ax10.set_xlabel('Loyalty Years', fontsize=9)
ax10.set_ylabel('Spending Score', fontsize=9)
style_ax(ax10, '⑩ Loyalty Years vs Spending Score')

plt.savefig('/home/claude/data_cleaning_project/dashboard.png',
            dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close()
print("\n✅ Dashboard saved!")
print("\nKey Insights:")
print(f"  • Dataset cleaned: {len(df_raw)} → {len(df)} rows ({removed_dupes} duplicates removed)")
print(f"  • Invalid income values fixed: {invalid_income.sum()}")
print(f"  • High-value customers: {df['High_Value'].sum()} ({df['High_Value'].mean()*100:.1f}%)")
print(f"  • Avg satisfaction: {df['Satisfaction_Rating'].mean():.2f}/5.0")
print(f"  • Income-Spending correlation: {df['Annual_Income_K'].corr(df['Spending_Score']):.3f}")
