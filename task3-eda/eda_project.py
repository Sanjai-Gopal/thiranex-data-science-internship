import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2024)
n = 800

# ── Generate Student Performance Dataset ──
departments = ['Computer Science','Electronics','Mechanical','Civil','Business']
data = {
    'StudentID': [f'STU{1000+i}' for i in range(n)],
    'Age': np.random.randint(18, 26, n),
    'Gender': np.random.choice(['Male','Female'], n, p=[0.55,0.45]),
    'Department': np.random.choice(departments, n, p=[0.3,0.2,0.2,0.15,0.15]),
    'Study_Hours_Per_Day': np.round(np.random.normal(4.5, 1.5, n).clip(0,12), 1),
    'Attendance_Pct': np.round(np.random.beta(7,2,n)*100, 1),
    'Assignment_Score': np.round(np.random.normal(72, 12, n).clip(0,100), 1),
    'Midterm_Score': np.round(np.random.normal(68, 15, n).clip(0,100), 1),
    'Final_Score': np.round(np.random.normal(70, 14, n).clip(0,100), 1),
    'Projects_Completed': np.random.randint(0, 6, n),
    'Part_Time_Job': np.random.choice([0,1], n, p=[0.65,0.35]),
    'Internet_Hours': np.round(np.random.normal(3.5, 1.2, n).clip(0,10), 1),
    'Sleep_Hours': np.round(np.random.normal(7, 1.2, n).clip(4,10), 1),
    'Family_Income_LPA': np.round(np.random.lognormal(1.5, 0.8, n).clip(1,50), 1),
}
df = pd.DataFrame(data)

# Make Final_Score correlate with study hours + attendance
df['Final_Score'] = (
    0.35 * df['Study_Hours_Per_Day'] * 6 +
    0.25 * df['Attendance_Pct'] * 0.7 +
    0.2  * df['Midterm_Score'] +
    0.2  * df['Assignment_Score'] +
    np.random.normal(0, 6, n)
).clip(0, 100).round(1)

df['GPA'] = (df['Final_Score'] / 100 * 4).clip(0,4).round(2)
df['Grade'] = pd.cut(df['Final_Score'],
    bins=[0,40,55,65,75,85,100],
    labels=['F','D','C','B','A','A+'])
df['Pass'] = (df['Final_Score'] >= 50).astype(int)

df.to_csv('/home/claude/task3_eda/student_performance.csv', index=False)
print("Dataset created:", df.shape)

# ── DASHBOARD ──
BG='#0a0e1a'; PANEL='#111827'; TEXT='#f1f5f9'; MUTED='#64748b'
C1='#38bdf8'; C2='#f472b6'; C3='#34d399'; C4='#fb923c'; C5='#a78bfa'; C6='#fbbf24'
COLS = [C1,C2,C3,C4,C5,C6]

plt.style.use('dark_background')
fig = plt.figure(figsize=(22,18), facecolor=BG)
gs = gridspec.GridSpec(3,4, figure=fig, hspace=0.48, wspace=0.38,
                       left=0.05,right=0.97,top=0.92,bottom=0.06)

def sax(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor('#1e293b')
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=10)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)

fig.text(0.5,0.965,'Student Performance — Exploratory Data Analysis',
         ha='center',fontsize=21,fontweight='bold',color=TEXT,fontfamily='monospace')
fig.text(0.5,0.94,'Thiranex Data Science Internship  •  Sanjai G  •  Task 3',
         ha='center',fontsize=11,color=MUTED)

# 1. Final Score Distribution
ax1=fig.add_subplot(gs[0,0])
ax1.hist(df['Final_Score'],bins=30,color=C1,alpha=0.85,edgecolor=BG,lw=0.5)
ax1.axvline(df['Final_Score'].mean(),color=C4,lw=2,linestyle='--',label=f"Mean {df['Final_Score'].mean():.1f}")
ax1.axvline(df['Final_Score'].median(),color=C2,lw=2,linestyle=':',label=f"Median {df['Final_Score'].median():.1f}")
ax1.legend(fontsize=8,facecolor=PANEL,edgecolor=MUTED,labelcolor=TEXT)
ax1.set_xlabel('Final Score',fontsize=9); ax1.set_ylabel('Count',fontsize=9)
sax(ax1,'① Final Score Distribution')

# 2. Grade breakdown
ax2=fig.add_subplot(gs[0,1])
gc=df['Grade'].value_counts().sort_index()
bars=ax2.bar(gc.index,gc.values,color=COLS[:len(gc)],alpha=0.85,edgecolor=BG,lw=1.5)
for b,v in zip(bars,gc.values):
    ax2.text(b.get_x()+b.get_width()/2,b.get_height()+3,str(v),
             ha='center',color=TEXT,fontsize=9,fontweight='bold')
ax2.set_ylabel('Count',fontsize=9)
sax(ax2,'② Grade Distribution')

# 3. Study Hours vs Final Score
ax3=fig.add_subplot(gs[0,2])
sc=ax3.scatter(df['Study_Hours_Per_Day'],df['Final_Score'],
               c=df['Attendance_Pct'],cmap='viridis',alpha=0.5,s=15)
cb=plt.colorbar(sc,ax=ax3); cb.ax.tick_params(colors=MUTED,labelsize=7)
cb.set_label('Attendance %',color=MUTED,fontsize=8)
z=np.polyfit(df['Study_Hours_Per_Day'],df['Final_Score'],1)
xl=np.linspace(0,12,100)
ax3.plot(xl,np.poly1d(z)(xl),color=C4,lw=2,linestyle='--')
ax3.set_xlabel('Study Hours/Day',fontsize=9); ax3.set_ylabel('Final Score',fontsize=9)
r,p=stats.pearsonr(df['Study_Hours_Per_Day'],df['Final_Score'])
ax3.set_title(f'③ Study Hours vs Final Score  (r={r:.2f})',color=TEXT,fontsize=10,fontweight='bold',pad=10)
ax3.set_facecolor(PANEL)
for s in ax3.spines.values(): s.set_edgecolor('#1e293b')
ax3.tick_params(colors=MUTED,labelsize=9)

# 4. Dept-wise GPA boxplot
ax4=fig.add_subplot(gs[0,3])
dept_short={'Computer Science':'CS','Electronics':'ECE','Mechanical':'MECH','Civil':'CIVIL','Business':'BUS'}
df['Dept_S']=df['Department'].map(dept_short)
order=[dept_short[d] for d in departments]
bp=ax4.boxplot([df[df['Dept_S']==d]['GPA'].values for d in order],
               labels=order,patch_artist=True,
               medianprops={'color':BG,'linewidth':2})
for patch,col in zip(bp['boxes'],COLS):
    patch.set_facecolor(col); patch.set_alpha(0.75)
for el in ['whiskers','caps','fliers']:
    for it in bp[el]: it.set_color(MUTED)
ax4.set_xticklabels(order,fontsize=7.5)
ax4.set_ylabel('GPA',fontsize=9)
sax(ax4,'④ GPA by Department')

# 5. Attendance vs Final (with regression bands)
ax5=fig.add_subplot(gs[1,0:2])
for gender,color in zip(['Male','Female'],[C1,C2]):
    sub=df[df['Gender']==gender]
    ax5.scatter(sub['Attendance_Pct'],sub['Final_Score'],alpha=0.4,s=12,color=color,label=gender)
    z2=np.polyfit(sub['Attendance_Pct'],sub['Final_Score'],1)
    xl2=np.linspace(sub['Attendance_Pct'].min(),sub['Attendance_Pct'].max(),100)
    ax5.plot(xl2,np.poly1d(z2)(xl2),color=color,lw=2)
ax5.legend(fontsize=9,facecolor=PANEL,edgecolor=MUTED,labelcolor=TEXT)
ax5.set_xlabel('Attendance %',fontsize=9); ax5.set_ylabel('Final Score',fontsize=9)
r2,_=stats.pearsonr(df['Attendance_Pct'],df['Final_Score'])
sax(ax5,f'⑤ Attendance vs Final Score by Gender  (r={r2:.2f})')

# 6. Correlation heatmap
ax6=fig.add_subplot(gs[1,2])
num_cols=['Study_Hours_Per_Day','Attendance_Pct','Assignment_Score',
          'Midterm_Score','Final_Score','Sleep_Hours','Internet_Hours']
corr=df[num_cols].corr()
sns.heatmap(corr,ax=ax6,annot=True,fmt='.2f',cmap='coolwarm',center=0,
            annot_kws={'size':7},linewidths=0.5,linecolor=BG,
            mask=np.triu(np.ones_like(corr,dtype=bool)),
            cbar_kws={'shrink':0.8})
ax6.tick_params(labelsize=7,colors=MUTED); ax6.set_facecolor(PANEL)
ax6.set_title('⑥ Feature Correlation Heatmap',color=TEXT,fontsize=10,fontweight='bold',pad=10)

# 7. Part-time job impact
ax7=fig.add_subplot(gs[1,3])
no_job=df[df['Part_Time_Job']==0]['Final_Score']
with_job=df[df['Part_Time_Job']==1]['Final_Score']
bp2=ax7.boxplot([no_job,with_job],labels=['No Job','Part-Time Job'],
                patch_artist=True,medianprops={'color':BG,'linewidth':2})
bp2['boxes'][0].set_facecolor(C3); bp2['boxes'][0].set_alpha(0.8)
bp2['boxes'][1].set_facecolor(C4); bp2['boxes'][1].set_alpha(0.8)
for el in ['whiskers','caps']: 
    for it in bp2[el]: it.set_color(MUTED)
ax7.set_ylabel('Final Score',fontsize=9)
t,p_val=stats.ttest_ind(no_job,with_job)
ax7.set_title(f'⑦ Part-Time Job Impact\n(t={t:.2f}, p={p_val:.3f})',
              color=TEXT,fontsize=10,fontweight='bold',pad=6)
ax7.set_facecolor(PANEL)
for s in ax7.spines.values(): s.set_edgecolor('#1e293b')
ax7.tick_params(colors=MUTED,labelsize=9)

# 8. Sleep vs Performance
ax8=fig.add_subplot(gs[2,0])
bins_sleep=pd.cut(df['Sleep_Hours'],bins=[3,5,6,7,8,10],labels=['<5h','5-6h','6-7h','7-8h','>8h'])
sleep_perf=df.groupby(bins_sleep,observed=True)['Final_Score'].mean()
ax8.bar(sleep_perf.index,sleep_perf.values,color=COLS,alpha=0.85,edgecolor=BG,lw=1.5)
for i,(idx,v) in enumerate(sleep_perf.items()):
    ax8.text(i,v+0.5,f'{v:.1f}',ha='center',color=TEXT,fontsize=9,fontweight='bold')
ax8.set_ylabel('Avg Final Score',fontsize=9); ax8.set_ylim(0,100)
sax(ax8,'⑧ Sleep Hours vs Avg Score')

# 9. Pass/Fail by department
ax9=fig.add_subplot(gs[2,1])
pf=df.groupby(['Dept_S','Pass'],observed=True).size().unstack(fill_value=0)
pf_pct=pf.div(pf.sum(axis=1),axis=0)*100
pf_pct.plot(kind='bar',ax=ax9,color=[C4,C3],edgecolor=BG,lw=1)
ax9.set_xticklabels(ax9.get_xticklabels(),rotation=30,ha='right',fontsize=8)
ax9.set_ylabel('Percentage (%)',fontsize=9)
ax9.legend(['Fail','Pass'],fontsize=8,facecolor=PANEL,edgecolor=MUTED,labelcolor=TEXT)
sax(ax9,'⑨ Pass Rate by Department')

# 10. Projects vs GPA
ax10=fig.add_subplot(gs[2,2])
proj_gpa=df.groupby('Projects_Completed')['GPA'].agg(['mean','std','count'])
ax10.bar(proj_gpa.index,proj_gpa['mean'],
         yerr=proj_gpa['std']/np.sqrt(proj_gpa['count']),
         color=C5,alpha=0.85,edgecolor=BG,lw=1.5,
         error_kw={'color':TEXT,'capsize':5,'lw':1.5})
ax10.set_xlabel('Projects Completed',fontsize=9); ax10.set_ylabel('Avg GPA',fontsize=9)
sax(ax10,'⑩ Projects Completed vs Avg GPA')

# 11. Family Income vs Score violin
ax11=fig.add_subplot(gs[2,3])
df['Income_Tier']=pd.cut(df['Family_Income_LPA'],bins=[0,3,8,20,60],
                          labels=['Low\n(<3L)','Mid\n(3-8L)','High\n(8-20L)','Very High\n(>20L)'])
income_groups=[df[df['Income_Tier']==t]['Final_Score'].values 
               for t in ['Low\n(<3L)','Mid\n(3-8L)','High\n(8-20L)','Very High\n(>20L)']]
vp=ax11.violinplot(income_groups,positions=[1,2,3,4],showmedians=True)
for i,(body,color) in enumerate(zip(vp['bodies'],COLS)):
    body.set_facecolor(color); body.set_alpha(0.7); body.set_edgecolor(BG)
vp['cmedians'].set_color(TEXT); vp['cmedians'].set_lw(2)
vp['cbars'].set_color(MUTED); vp['cmaxes'].set_color(MUTED); vp['cmins'].set_color(MUTED)
ax11.set_xticks([1,2,3,4])
ax11.set_xticklabels(['Low\n(<3L)','Mid\n(3-8L)','High\n(8-20L)','Very High\n(>20L)'],fontsize=8)
ax11.set_ylabel('Final Score',fontsize=9)
sax(ax11,'⑪ Score Distribution by Family Income')

plt.savefig('/home/claude/task3_eda/eda_dashboard.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close()
print("✅ EDA Dashboard saved!")

# Print key stats
print(f"\n📊 KEY INSIGHTS:")
print(f"  Pass Rate: {df['Pass'].mean()*100:.1f}%")
print(f"  Study-Score correlation: {stats.pearsonr(df['Study_Hours_Per_Day'],df['Final_Score'])[0]:.3f}")
print(f"  Attendance-Score correlation: {stats.pearsonr(df['Attendance_Pct'],df['Final_Score'])[0]:.3f}")
print(f"  Top dept by GPA: {df.groupby('Department')['GPA'].mean().idxmax()}")
print(f"  Avg GPA: {df['GPA'].mean():.2f}/4.0")
