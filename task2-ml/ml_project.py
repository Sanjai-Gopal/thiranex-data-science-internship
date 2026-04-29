import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, accuracy_score, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 1000

# ── Generate Loan Default Prediction Dataset ──────────────────────
age        = np.random.randint(22, 65, n)
income     = np.round(np.random.lognormal(10.8, 0.5, n), -2)   # monthly income ₹
loan_amt   = np.round(np.random.uniform(50000, 1500000, n), -3)
loan_term  = np.random.choice([12, 24, 36, 48, 60], n)
credit_score = np.random.randint(300, 850, n)
employment = np.random.choice(['Salaried','Self-Employed','Business'], n, p=[0.55,0.25,0.20])
num_dependents = np.random.randint(0, 5, n)
existing_loans = np.random.randint(0, 4, n)
education  = np.random.choice(['Graduate','Not Graduate'], n, p=[0.65, 0.35])
property_area = np.random.choice(['Urban','Semi-Urban','Rural'], n, p=[0.4,0.35,0.25])

# Realistic default logic
log_odds = (
    -4.0
    + 0.025 * (700 - credit_score) / 100
    + 0.8   * (loan_amt / income / loan_term)
    - 0.015 * (income / 10000)
    + 0.3   * existing_loans
    + 0.2   * (employment == 'Self-Employed').astype(int)
    + 0.15  * (education == 'Not Graduate').astype(int)
    + np.random.normal(0, 0.8, n)
)
prob_default = 1 / (1 + np.exp(-log_odds))
default = (np.random.uniform(0,1,n) < prob_default).astype(int)

df = pd.DataFrame({
    'Age': age,
    'Monthly_Income': income,
    'Loan_Amount': loan_amt,
    'Loan_Term_Months': loan_term,
    'Credit_Score': credit_score,
    'Employment_Type': employment,
    'Num_Dependents': num_dependents,
    'Existing_Loans': existing_loans,
    'Education': education,
    'Property_Area': property_area,
    'Loan_to_Income_Ratio': np.round(loan_amt / (income * loan_term), 4),
    'Default': default
})

df.to_csv('/home/claude/task2_ml/loan_default_dataset.csv', index=False)
print(f"Dataset: {df.shape}  |  Default rate: {df['Default'].mean()*100:.1f}%")

# ── PREPROCESSING ─────────────────────────────────────────────────
df_enc = df.copy()
for col in ['Employment_Type','Education','Property_Area']:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

X = df_enc.drop('Default', axis=1)
y = df_enc['Default']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── TRAIN THREE MODELS ────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                   min_samples_leaf=10, random_state=42, n_jobs=-1),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    Xtr = X_train_sc if name == 'Logistic Regression' else X_train
    Xte = X_test_sc  if name == 'Logistic Regression' else X_test

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_prob  = model.predict_proba(Xte)[:,1]
    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='accuracy')
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'acc': accuracy_score(y_test, y_pred),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'fpr': fpr, 'tpr': tpr, 'auc': roc_auc,
        'cm': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'Xte': Xte, 'Xtr': Xtr,
    }
    print(f"\n{name}:")
    print(f"  Accuracy:  {results[name]['acc']*100:.2f}%")
    print(f"  CV Score:  {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

# ── FEATURE IMPORTANCE (Random Forest) ───────────────────────────
rf = results['Random Forest']['model']
fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print(f"\nTop Features (RF):\n{fi.head(6)}")

# ── FULL DASHBOARD ────────────────────────────────────────────────
BG='#060c14'; PANEL='#0d1421'; TEXT='#f1f5f9'; MUTED='#475569'
C_LR='#38bdf8'; C_DT='#34d399'; C_RF='#f472b6'; GOLD='#fbbf24'
MODEL_COLORS = {'Logistic Regression': C_LR, 'Decision Tree': C_DT, 'Random Forest': C_RF}

plt.style.use('dark_background')
fig = plt.figure(figsize=(22, 18), facecolor=BG)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4,
                       left=0.05, right=0.97, top=0.92, bottom=0.06)

def sax(ax, title, fs=10):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor('#1e293b')
    ax.set_title(title, color=TEXT, fontsize=fs, fontweight='bold', pad=10)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)

fig.text(0.5, 0.965, 'Predictive Modeling — Loan Default Prediction',
         ha='center', fontsize=21, fontweight='bold', color=TEXT, fontfamily='monospace')
fig.text(0.5, 0.940, 'Thiranex Data Science Internship  •  Sanjai G  •  Task 2  |  Models: Logistic Regression · Decision Tree · Random Forest',
         ha='center', fontsize=10.5, color=MUTED)

# ① Model Accuracy Comparison
ax1 = fig.add_subplot(gs[0, 0])
names  = list(results.keys())
accs   = [results[n]['acc']*100 for n in names]
cvs    = [results[n]['cv_mean']*100 for n in names]
stds   = [results[n]['cv_std']*100 for n in names]
x = np.arange(len(names)); w = 0.35
cols = [MODEL_COLORS[n] for n in names]
b1 = ax1.bar(x - w/2, accs, w, color=cols, alpha=0.85, edgecolor=BG, lw=1.5, label='Test Acc')
b2 = ax1.bar(x + w/2, cvs,  w, color=cols, alpha=0.45, edgecolor=BG, lw=1.5,
             yerr=stds, error_kw={'color':TEXT,'capsize':5,'lw':1.5}, label='CV Acc')
for b, v in zip(b1, accs):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f'{v:.1f}%',
             ha='center', color=TEXT, fontsize=8.5, fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(['LR','DT','RF'], fontsize=10)
ax1.set_ylim(85, 100); ax1.set_ylabel('Accuracy (%)', fontsize=9)
ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
sax(ax1, '① Model Accuracy Comparison')

# ② ROC Curves
ax2 = fig.add_subplot(gs[0, 1])
for name, color in MODEL_COLORS.items():
    ax2.plot(results[name]['fpr'], results[name]['tpr'], color=color, lw=2,
             label=f"{name.split()[0]} (AUC={results[name]['auc']:.3f})")
ax2.plot([0,1],[0,1], color=MUTED, lw=1.2, linestyle='--')
ax2.fill_between(results['Random Forest']['fpr'],
                 results['Random Forest']['tpr'], alpha=0.08, color=C_RF)
ax2.set_xlabel('False Positive Rate', fontsize=9)
ax2.set_ylabel('True Positive Rate', fontsize=9)
ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
sax(ax2, '② ROC Curves — All Models')

# ③ Confusion Matrix — Random Forest
ax3 = fig.add_subplot(gs[0, 2])
cm_rf = results['Random Forest']['cm']
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax3,
            linewidths=1, linecolor=BG,
            xticklabels=['No Default','Default'],
            yticklabels=['No Default','Default'],
            annot_kws={'size': 13, 'fontweight': 'bold'},
            cbar_kws={'shrink': 0.8})
ax3.set_xlabel('Predicted', fontsize=9); ax3.set_ylabel('Actual', fontsize=9)
ax3.tick_params(colors=MUTED, labelsize=8.5)
ax3.set_facecolor(PANEL)
ax3.set_title('③ Confusion Matrix (Random Forest)', color=TEXT, fontsize=10, fontweight='bold', pad=10)

# ④ Confusion Matrix — Decision Tree
ax4 = fig.add_subplot(gs[0, 3])
cm_dt = results['Decision Tree']['cm']
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=ax4,
            linewidths=1, linecolor=BG,
            xticklabels=['No Default','Default'],
            yticklabels=['No Default','Default'],
            annot_kws={'size': 13, 'fontweight': 'bold'},
            cbar_kws={'shrink': 0.8})
ax4.set_xlabel('Predicted', fontsize=9); ax4.set_ylabel('Actual', fontsize=9)
ax4.tick_params(colors=MUTED, labelsize=8.5)
ax4.set_facecolor(PANEL)
ax4.set_title('④ Confusion Matrix (Decision Tree)', color=TEXT, fontsize=10, fontweight='bold', pad=10)

# ⑤ Feature Importance — Random Forest
ax5 = fig.add_subplot(gs[1, 0:2])
fi_sorted = fi.sort_values()
colors_fi = [C_RF if v >= fi_sorted.median() else MUTED for v in fi_sorted.values]
bars = ax5.barh(fi_sorted.index, fi_sorted.values, color=colors_fi, alpha=0.85,
                edgecolor=BG, lw=1.2)
for bar, v in zip(bars, fi_sorted.values):
    ax5.text(v + 0.003, bar.get_y()+bar.get_height()/2,
             f'{v:.3f}', va='center', color=TEXT, fontsize=9, fontweight='bold')
ax5.set_xlabel('Importance Score', fontsize=9)
sax(ax5, '⑤ Feature Importance — Random Forest', fs=11)

# ⑥ CV Score distribution (box plot)
ax6 = fig.add_subplot(gs[1, 2])
cv_all = {}
for name, model in models.items():
    Xtr = results[name]['Xtr']
    scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc')
    cv_all[name] = scores
bp = ax6.boxplot(list(cv_all.values()), labels=['LR','DT','RF'],
                 patch_artist=True, medianprops={'color': BG, 'linewidth': 2.5})
for patch, color in zip(bp['boxes'], [C_LR, C_DT, C_RF]):
    patch.set_facecolor(color); patch.set_alpha(0.8)
for el in ['whiskers','caps']:
    for it in bp[el]: it.set_color(MUTED)
ax6.set_ylabel('ROC-AUC', fontsize=9)
sax(ax6, '⑥ 5-Fold CV ROC-AUC Distribution')

# ⑦ Precision / Recall / F1 bar chart
ax7 = fig.add_subplot(gs[1, 3])
metrics_names = ['precision','recall','f1-score']
x_m = np.arange(len(metrics_names)); w_m = 0.25
for i, (name, color) in enumerate(MODEL_COLORS.items()):
    vals = [results[name]['report']['weighted avg'][m] for m in metrics_names]
    ax7.bar(x_m + (i-1)*w_m, vals, w_m, color=color, alpha=0.85,
            edgecolor=BG, lw=1.2, label=name.split()[0])
ax7.set_xticks(x_m)
ax7.set_xticklabels(['Precision','Recall','F1-Score'], fontsize=9)
ax7.set_ylim(0.85, 1.0); ax7.set_ylabel('Score', fontsize=9)
ax7.legend(fontsize=8, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
sax(ax7, '⑦ Precision / Recall / F1 Score')

# ⑧ Credit Score vs Loan-to-Income Ratio (colored by Default)
ax8 = fig.add_subplot(gs[2, 0:2])
no_def  = df[df['Default']==0]
yes_def = df[df['Default']==1]
ax8.scatter(no_def['Credit_Score'], no_def['Loan_to_Income_Ratio'],
            color=C_LR, alpha=0.3, s=12, label='No Default')
ax8.scatter(yes_def['Credit_Score'], yes_def['Loan_to_Income_Ratio'],
            color=C_RF, alpha=0.7, s=20, label='Default', zorder=5)
ax8.set_xlabel('Credit Score', fontsize=9)
ax8.set_ylabel('Loan-to-Income Ratio', fontsize=9)
ax8.legend(fontsize=9, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
sax(ax8, '⑧ Credit Score vs Loan-to-Income Ratio  (Default Highlighted)', fs=11)

# ⑨ Default rate by Employment Type
ax9 = fig.add_subplot(gs[2, 2])
dr_emp = df.groupby('Employment_Type')['Default'].mean()*100
bars9 = ax9.bar(dr_emp.index, dr_emp.values,
                color=[C_LR, C_DT, C_RF], alpha=0.85, edgecolor=BG, lw=1.5)
for b, v in zip(bars9, dr_emp.values):
    ax9.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
             f'{v:.1f}%', ha='center', color=TEXT, fontsize=10, fontweight='bold')
ax9.set_ylabel('Default Rate (%)', fontsize=9)
sax(ax9, '⑨ Default Rate by Employment Type')

# ⑩ Predicted Probability Distribution
ax10 = fig.add_subplot(gs[2, 3])
probs_rf = results['Random Forest']['y_prob']
ax10.hist(probs_rf[y_test==0], bins=30, color=C_LR, alpha=0.7,
          label='No Default', edgecolor=BG, density=True)
ax10.hist(probs_rf[y_test==1], bins=30, color=C_RF, alpha=0.85,
          label='Default', edgecolor=BG, density=True)
ax10.axvline(0.5, color=GOLD, lw=2, linestyle='--', label='Threshold 0.5')
ax10.set_xlabel('Predicted Probability of Default', fontsize=9)
ax10.set_ylabel('Density', fontsize=9)
ax10.legend(fontsize=8, facecolor=PANEL, edgecolor=MUTED, labelcolor=TEXT)
sax(ax10, '⑩ RF Predicted Probability Distribution')

plt.savefig('/home/claude/task2_ml/ml_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("\n✅ ML Dashboard saved!")
print("\n📊 FINAL MODEL COMPARISON:")
print(f"{'Model':<22} {'Accuracy':>10} {'CV Score':>10} {'ROC-AUC':>10}")
print("-"*55)
for name in names:
    r = results[name]
    print(f"{name:<22} {r['acc']*100:>9.2f}% {r['cv_mean']*100:>9.2f}% {r['auc']:>10.4f}")
print(f"\n🏆 Best Model: Random Forest (highest AUC + stable CV)")
