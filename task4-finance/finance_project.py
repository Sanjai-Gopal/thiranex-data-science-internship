import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(99)

# ── Generate Realistic Stock Market Dataset (5 companies, 2 years) ──
companies = {
    'TECHCO':  {'base': 1200, 'vol': 0.022, 'trend': 0.0004},
    'BANKX':   {'base': 450,  'vol': 0.015, 'trend': 0.0002},
    'PHARMA':  {'base': 780,  'vol': 0.018, 'trend': 0.0003},
    'ENERGYCO':{'base': 320,  'vol': 0.025, 'trend': 0.0001},
    'RETAILCO':{'base': 560,  'vol': 0.020, 'trend': 0.00025},
}

dates = pd.date_range('2023-01-01', '2024-12-31', freq='B')
all_data = []

for ticker, params in companies.items():
    price = params['base']
    prices, volumes, opens, highs, lows = [], [], [], [], []
    
    # Add market events (crashes/rallies)
    crash_day = np.random.randint(100, 300)
    rally_day = np.random.randint(350, 480)
    
    for i, date in enumerate(dates):
        shock = 1.0
        if abs(i - crash_day) < 5: shock = 0.97
        if abs(i - rally_day) < 5: shock = 1.03
        
        daily_return = (np.random.normal(params['trend'], params['vol']) + 
                        np.log(shock)) 
        price = price * np.exp(daily_return)
        
        open_p = price * (1 + np.random.normal(0, 0.003))
        high_p = price * (1 + abs(np.random.normal(0, 0.008)))
        low_p  = price * (1 - abs(np.random.normal(0, 0.008)))
        vol    = int(np.random.lognormal(14, 0.5))
        
        prices.append(round(price, 2))
        opens.append(round(open_p, 2))
        highs.append(round(high_p, 2))
        lows.append(round(low_p, 2))
        volumes.append(vol)
    
    sector_map = {'TECHCO':'Technology','BANKX':'Finance','PHARMA':'Healthcare',
                  'ENERGYCO':'Energy','RETAILCO':'Retail'}
    tmp = pd.DataFrame({
        'Date': dates, 'Ticker': ticker, 'Sector': sector_map[ticker],
        'Open': opens, 'High': highs, 'Low': lows, 'Close': prices, 'Volume': volumes
    })
    tmp['Daily_Return'] = tmp['Close'].pct_change()
    tmp['MA_20'] = tmp['Close'].rolling(20).mean()
    tmp['MA_50'] = tmp['Close'].rolling(50).mean()
    tmp['Volatility_20'] = tmp['Daily_Return'].rolling(20).std() * np.sqrt(252)
    tmp['Cum_Return'] = (tmp['Close'] / tmp['Close'].iloc[0] - 1) * 100
    all_data.append(tmp)

df = pd.concat(all_data, ignore_index=True)
df.to_csv('/home/claude/task4_realworld/stock_market_data.csv', index=False)
print("Dataset created:", df.shape)

# ── DASHBOARD ──
BG='#060b14'; PANEL='#0d1421'; TEXT='#e2e8f0'; MUTED='#475569'
COLS = {'TECHCO':'#38bdf8','BANKX':'#34d399','PHARMA':'#f472b6',
        'ENERGYCO':'#fb923c','RETAILCO':'#a78bfa'}

plt.style.use('dark_background')
fig = plt.figure(figsize=(22,18), facecolor=BG)
gs = gridspec.GridSpec(3,4,figure=fig,hspace=0.5,wspace=0.38,
                       left=0.05,right=0.97,top=0.92,bottom=0.06)

def sax(ax, title, fs=10):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED,labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor('#1e293b')
    ax.set_title(title,color=TEXT,fontsize=fs,fontweight='bold',pad=10)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)

fig.text(0.5,0.965,'Stock Market Analysis — Finance Domain',
         ha='center',fontsize=21,fontweight='bold',color=TEXT,fontfamily='monospace')
fig.text(0.5,0.94,'Thiranex Data Science Internship  •  Sanjai G  •  Task 4',
         ha='center',fontsize=11,color=MUTED)

# 1. Price trends all companies
ax1=fig.add_subplot(gs[0,0:2])
for ticker, color in COLS.items():
    sub=df[df['Ticker']==ticker].set_index('Date')
    ax1.plot(sub.index, sub['Close'], color=color, lw=1.5, label=ticker, alpha=0.9)
ax1.legend(fontsize=8.5,facecolor=PANEL,edgecolor=MUTED,labelcolor=TEXT,ncol=5)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.get_xticklabels(),rotation=30,ha='right',fontsize=8)
ax1.set_ylabel('Price (₹)',fontsize=9)
sax(ax1,'① Stock Price Trends (2023–2024)',fs=11)

# 2. Cumulative returns
ax2=fig.add_subplot(gs[0,2:4])
for ticker, color in COLS.items():
    sub=df[df['Ticker']==ticker].set_index('Date')
    ax2.plot(sub.index,sub['Cum_Return'],color=color,lw=1.5,label=ticker)
ax2.axhline(0,color=MUTED,lw=1,linestyle='--')
ax2.fill_between(sub.index,[0]*len(sub),alpha=0.03,color='white')
ax2.legend(fontsize=8.5,facecolor=PANEL,edgecolor=MUTED,labelcolor=TEXT,ncol=5)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.get_xticklabels(),rotation=30,ha='right',fontsize=8)
ax2.set_ylabel('Cumulative Return (%)',fontsize=9)
sax(ax2,'② Cumulative Returns Comparison',fs=11)

# 3. Daily return distribution
ax3=fig.add_subplot(gs[1,0])
for ticker, color in COLS.items():
    sub=df[df['Ticker']==ticker]['Daily_Return'].dropna()
    ax3.hist(sub,bins=40,alpha=0.5,color=color,edgecolor='none',density=True,label=ticker)
ax3.set_xlabel('Daily Return',fontsize=9)
ax3.set_ylabel('Density',fontsize=9)
ax3.legend(fontsize=7.5,facecolor=PANEL,edgecolor=MUTED,labelcolor=TEXT)
sax(ax3,'③ Daily Return Distribution')

# 4. Volatility over time (TECHCO with MA)
ax4=fig.add_subplot(gs[1,1])
tech=df[df['Ticker']=='TECHCO'].set_index('Date')
ax4.plot(tech.index,tech['Close'],color=COLS['TECHCO'],lw=1.2,alpha=0.8,label='Price')
ax4.plot(tech.index,tech['MA_20'],color='#fbbf24',lw=1.5,linestyle='--',label='MA-20')
ax4.plot(tech.index,tech['MA_50'],color='#f472b6',lw=1.5,linestyle='-.',label='MA-50')
ax4.legend(fontsize=8,facecolor=PANEL,edgecolor=MUTED,labelcolor=TEXT)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
plt.setp(ax4.get_xticklabels(),rotation=30,ha='right',fontsize=7.5)
ax4.set_ylabel('Price (₹)',fontsize=9)
sax(ax4,'④ TECHCO: Price + Moving Averages')

# 5. Annualized Volatility comparison
ax5=fig.add_subplot(gs[1,2])
avg_vol=df.groupby('Ticker')['Volatility_20'].mean().sort_values(ascending=False)
colors_sorted=[COLS[t] for t in avg_vol.index]
bars=ax5.bar(avg_vol.index,avg_vol.values*100,color=colors_sorted,alpha=0.85,edgecolor=BG,lw=1.5)
for b,v in zip(bars,avg_vol.values*100):
    ax5.text(b.get_x()+b.get_width()/2,b.get_height()+0.2,f'{v:.1f}%',
             ha='center',color=TEXT,fontsize=9,fontweight='bold')
ax5.set_ylabel('Avg Annualized Volatility (%)',fontsize=9)
sax(ax5,'⑤ Volatility by Company')

# 6. Correlation matrix of returns
ax6=fig.add_subplot(gs[1,3])
pivot=df.pivot_table(index='Date',columns='Ticker',values='Daily_Return')
corr=pivot.corr()
sns.heatmap(corr,ax=ax6,annot=True,fmt='.2f',cmap='RdYlGn',center=0,
            annot_kws={'size':9},linewidths=0.5,linecolor=BG,
            cbar_kws={'shrink':0.8})
ax6.tick_params(labelsize=8.5,colors=MUTED); ax6.set_facecolor(PANEL)
ax6.set_title('⑥ Return Correlations Between Stocks',color=TEXT,fontsize=10,fontweight='bold',pad=10)

# 7. Monthly average return heatmap (TECHCO)
ax7=fig.add_subplot(gs[2,0:2])
tech2=df[df['Ticker']=='TECHCO'].copy()
tech2['Month']=tech2['Date'].dt.month_name().str[:3]
tech2['Year']=tech2['Date'].dt.year
monthly=tech2.groupby(['Year','Month'])['Daily_Return'].mean().unstack()
month_order=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthly=monthly[[m for m in month_order if m in monthly.columns]]
sns.heatmap(monthly*100,ax=ax7,annot=True,fmt='.2f',cmap='RdYlGn',center=0,
            annot_kws={'size':9},linewidths=0.8,linecolor=BG,
            cbar_kws={'shrink':0.6})
ax7.tick_params(labelsize=9,colors=MUTED); ax7.set_facecolor(PANEL)
ax7.set_ylabel('Year',fontsize=9)
ax7.set_title('⑦ TECHCO: Monthly Avg Daily Return (%) Heatmap',
              color=TEXT,fontsize=10,fontweight='bold',pad=10)

# 8. Risk vs Return scatter
ax8=fig.add_subplot(gs[2,2])
for ticker, color in COLS.items():
    sub=df[df['Ticker']==ticker]['Daily_Return'].dropna()
    annual_ret=sub.mean()*252*100
    annual_vol=sub.std()*np.sqrt(252)*100
    ax8.scatter(annual_vol,annual_ret,color=color,s=180,zorder=5,edgecolors=TEXT,lw=1.5)
    ax8.annotate(ticker,(annual_vol,annual_ret),textcoords='offset points',
                 xytext=(6,4),fontsize=8.5,color=color,fontweight='bold')
ax8.set_xlabel('Annual Volatility (%)',fontsize=9)
ax8.set_ylabel('Annual Return (%)',fontsize=9)
ax8.axhline(0,color=MUTED,lw=0.8,linestyle='--')
sax(ax8,'⑧ Risk vs Return (Efficient Frontier)')

# 9. Volume analysis
ax9=fig.add_subplot(gs[2,3])
avg_vol_by=df.groupby('Ticker')['Volume'].mean()/1e6
colors_v=[COLS[t] for t in avg_vol_by.index]
bars2=ax9.bar(avg_vol_by.index,avg_vol_by.values,color=colors_v,alpha=0.85,edgecolor=BG,lw=1.5)
for b,v in zip(bars2,avg_vol_by.values):
    ax9.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{v:.2f}M',
             ha='center',color=TEXT,fontsize=9,fontweight='bold')
ax9.set_ylabel('Avg Daily Volume (Millions)',fontsize=9)
sax(ax9,'⑨ Average Trading Volume by Stock')

plt.savefig('/home/claude/task4_realworld/finance_dashboard.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close()

# Summary stats
print("✅ Finance Dashboard saved!")
final_returns={}
for ticker in companies:
    sub=df[df['Ticker']==ticker]
    ret=((sub['Close'].iloc[-1]/sub['Close'].iloc[0])-1)*100
    final_returns[ticker]=ret
    
print(f"\n📈 2-Year Total Returns:")
for t,r in sorted(final_returns.items(),key=lambda x:-x[1]):
    print(f"  {t}: {r:+.1f}%")
    
sharpe={}
for ticker in companies:
    sub=df[df['Ticker']==ticker]['Daily_Return'].dropna()
    sharpe[ticker]=(sub.mean()*252)/(sub.std()*np.sqrt(252))
print(f"\n📊 Sharpe Ratios:")
for t,s in sorted(sharpe.items(),key=lambda x:-x[1]):
    print(f"  {t}: {s:.3f}")
