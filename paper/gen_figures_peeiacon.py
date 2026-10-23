import numpy as np, matplotlib, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
matplotlib.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'legend.framealpha':0.92,'legend.borderpad':0.4,
    'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'savefig.pad_inches':0.04,'axes.linewidth':0.65,
    'grid.linewidth':0.35,'lines.linewidth':1.4,'lines.markersize':4.5,
    'axes.spines.top':False,'axes.spines.right':False,
})
os.makedirs('/home/claude/paper/figures',exist_ok=True)
CONDITIONS=['clean','loss\nlow','loss\nmed','loss\nhigh',
            'drift\nlow','drift\nmed','drift\nhigh',
            'delay\nlow','delay\nmed','delay\nhigh','severe']
FTRL_MS=[91.12,91.97,93.82,88.00,90.75,89.65,88.53,91.38,88.31,88.37,91.01]
MLP_MS= [89.08,88.06,87.57,89.81,91.17,89.31,90.00,89.95,90.39,92.14,89.73]
FTRL_TD=[2.826,2.256,3.029,1.777,2.164,2.298,1.312,1.429,1.765,1.135,1.916]
MLP_TD= [1.279,1.633,2.926,0.892,1.639,1.458,2.012,2.351,2.280,2.796,1.352]
FTRL_R=[-46.972,-47.115,-48.423,-44.888,-46.456,-45.973,-44.921,-46.405,-45.039,-44.752,-46.462]
MLP_R= [-45.180,-44.849,-45.250,-45.351,-46.405,-45.383,-46.007,-46.151,-46.334,-47.470,-45.540]
FC='#4C42A8'; MC='#CC4B25'; SC='#B07D15'; EC='#1A5C9E'; GC='#1D9E75'
x=np.arange(11); W=0.36
spans=[(-0.5,3.5,'#EFF5FF','Packet loss'),(3.5,6.5,'#EFFFEF','Sensor drift'),
       (6.5,9.5,'#FFF8EC','Bursty delay'),(9.5,10.5,'#FFECEC','Severe')]
def shade(ax,top,off=0):
    for x0,x1,fc,lb in spans:
        ax.axvspan(x0,x1,alpha=0.32,color=fc,zorder=0)
        ax.text((x0+x1)/2,top+off,lb,ha='center',va='bottom',fontsize=5.5,color='#555',style='italic')

# Fig2 Makespan
fig,ax=plt.subplots(figsize=(3.5,2.5))
ax.bar(x-W/2,FTRL_MS,W,color=FC,alpha=0.88,edgecolor='white',lw=0.3,label='FT-RLS',zorder=3)
ax.bar(x+W/2,MLP_MS, W,color=MC,alpha=0.88,edgecolor='white',lw=0.3,label='MLP-noisy',zorder=3)
ax.axhline(70.04,color=SC,lw=1.2,ls='--',label='SPT (70.0)',zorder=4)
ax.axhline(82.47,color=EC,lw=1.0,ls=':' ,label='EDD (82.5)',zorder=4)
shade(ax,101.5,0.5)
ax.set_xticks(x); ax.set_xticklabels(CONDITIONS,fontsize=5.8)
ax.set_ylabel('Mean makespan (time units)',labelpad=4); ax.set_ylim(60,103)
ax.yaxis.grid(True,ls=':',lw=0.35,zorder=0,alpha=0.7); ax.set_axisbelow(True)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=4,frameon=True,
          fontsize=6.5,handlelength=1.2,handletextpad=0.4,columnspacing=0.8,edgecolor='#ccc')
plt.tight_layout()
plt.savefig('/home/claude/paper/figures/fig2_makespan.pdf',bbox_inches='tight'); plt.close(); print('Fig2')

# Fig3 Tardiness
fig,ax=plt.subplots(figsize=(3.5,2.4))
ax.bar(x-W/2,FTRL_TD,W,color=FC,alpha=0.88,edgecolor='white',lw=0.3,label='FT-RLS',zorder=3)
ax.bar(x+W/2,MLP_TD, W,color=MC,alpha=0.88,edgecolor='white',lw=0.3,label='MLP-noisy',zorder=3)
shade(ax,3.38,0.04)
ax.set_xticks(x); ax.set_xticklabels(CONDITIONS,fontsize=5.8)
ax.set_ylabel('Mean weighted tardiness',labelpad=4); ax.set_ylim(0,3.55)
ax.yaxis.grid(True,ls=':',lw=0.35,zorder=0,alpha=0.7); ax.set_axisbelow(True)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=2,frameon=True,
          fontsize=6.5,handlelength=1.2,handletextpad=0.4,edgecolor='#ccc')
plt.tight_layout()
plt.savefig('/home/claude/paper/figures/fig3_tardiness.pdf',bbox_inches='tight'); plt.close(); print('Fig3')

# Fig4 Degradation
fig,axes=plt.subplots(1,3,figsize=(3.5,2.0),sharey=True); fig.subplots_adjust(wspace=0.08)
fams=[('Packet loss',[1,2,3],[1,2,3]),('Sensor drift',[1,2,3],[4,5,6]),('Bursty delay',[1,2,3],[7,8,9])]
fcs=['#3A7CC8',GC,FC]
for ax,(title,sev,idxs),fc in zip(axes,fams,fcs):
    ft=[FTRL_R[i] for i in idxs]; mlp=[MLP_R[i] for i in idxs]
    ax.plot(sev,ft,'o-',color=fc,lw=1.5,ms=4.5,zorder=4)
    ax.plot(sev,mlp,'s--',color=MC,lw=1.5,ms=4.5,zorder=4,alpha=0.85)
    ax.axhline(FTRL_R[0],color=fc,lw=0.6,ls=':',alpha=0.45)
    ax.axhline(MLP_R[0], color=MC,lw=0.6,ls=':',alpha=0.45)
    if title=='Bursty delay':
        ylo,yhi=min(ft[-1],mlp[-1]),max(ft[-1],mlp[-1])
        ax.annotate('',xy=(3,yhi),xytext=(3,ylo),
                    arrowprops=dict(arrowstyle='<->',color='#333',lw=0.9,mutation_scale=8))
        ax.text(3.12,(ylo+yhi)/2,f'$\\Delta${abs(mlp[-1]-ft[-1]):.2f}',
                va='center',fontsize=6,color='#333')
    ax.set_title(title,fontsize=7,pad=3,fontweight='bold')
    ax.set_xticks([1,2,3]); ax.set_xticklabels(['Low','Med','High'],fontsize=6.5)
    ax.set_xlabel('Noise severity',fontsize=7,labelpad=2)
    ax.yaxis.grid(True,ls=':',lw=0.35,alpha=0.7); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
axes[0].set_ylabel('Mean episode reward\n(higher = better)',fontsize=7,labelpad=3)
axes[0].set_ylim(-49.8,-43.2)
handles=[plt.Line2D([0],[0],color=FC,marker='o',lw=1.4,ms=4.5,label='FT-RLS'),
         plt.Line2D([0],[0],color=MC,marker='s',lw=1.4,ls='--',ms=4.5,label='MLP-noisy')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(0.5,-0.10),ncol=2,
           fontsize=7,frameon=True,handlelength=1.4,handletextpad=0.4,edgecolor='#ccc')
plt.savefig('/home/claude/paper/figures/fig4_degradation.pdf',bbox_inches='tight'); plt.close(); print('Fig4')

# Fig5 Heatmap
fig,ax=plt.subplots(figsize=(3.5,1.35))
matrix=np.array([FTRL_TD,MLP_TD])
im=ax.imshow(matrix,aspect='auto',cmap='RdYlGn_r',vmin=0,vmax=3.2)
cbar=plt.colorbar(im,ax=ax,fraction=0.035,pad=0.02)
cbar.ax.tick_params(labelsize=6); cbar.set_label('Tardiness',fontsize=6.5)
ax.set_xticks(range(11)); ax.set_xticklabels(CONDITIONS,fontsize=5.2)
ax.set_yticks([0,1]); ax.set_yticklabels(['FT-RLS','MLP-noisy'],fontsize=7)
for i in range(2):
    for j in range(11):
        val=matrix[i,j]; col='white' if val>2.1 else '#111'
        ax.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=5.2,color=col,fontweight='bold')
for j in range(11):
    if FTRL_TD[j]<MLP_TD[j]:
        rect=mpatches.FancyBboxPatch((j-0.485,-0.485),0.97,0.97,
            boxstyle='round,pad=0.02',fill=False,edgecolor=GC,lw=1.4,zorder=5)
        ax.add_patch(rect)
plt.tight_layout()
plt.savefig('/home/claude/paper/figures/fig5_heatmap.pdf',bbox_inches='tight'); plt.close(); print('Fig5')

# Fig6 Win-rate — FIXED: legend top-left, value labels above bars, no overlap
fig,axes=plt.subplots(1,2,figsize=(3.5,2.0)); fig.subplots_adjust(wspace=0.52)
cats=['Makespan','Tardiness','Reward']; wins=[7,8,7]; total=11; y=np.arange(3)
axes[0].barh(y,[total]*3,height=0.48,color='#E5E5E5',zorder=1)
axes[0].barh(y,wins,height=0.48,color=FC,alpha=0.87,zorder=2)
for yy,val in zip(y,wins):
    axes[0].text(val+0.2,yy,f'{val}/{total}',va='center',fontsize=7.5,fontweight='bold',color=FC)
axes[0].axvline(5.5,color='#AAA',lw=0.7,ls='--')
axes[0].set_yticks(y); axes[0].set_yticklabels(cats,fontsize=7.5)
axes[0].set_xlim(0,13.5); axes[0].set_xlabel('Win count (of 11)',fontsize=7,labelpad=3)
axes[0].set_title('Win rate',fontsize=7.5,fontweight='bold',pad=3)
axes[0].xaxis.grid(True,ls=':',lw=0.35,alpha=0.7,zorder=0); axes[0].set_axisbelow(True)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

fam_lbs=['Loss','Drift','Delay','Sev.']; idxs_f=[[1,2,3],[4,5,6],[7,8,9],[10]]
adv_ms=[np.mean([MLP_MS[i]-FTRL_MS[i] for i in idx]) for idx in idxs_f]
adv_td=[np.mean([MLP_TD[i]-FTRL_TD[i] for i in idx]) for idx in idxs_f]
x2=np.arange(4); w2=0.36
r1=axes[1].bar(x2-w2/2,adv_ms,w2,color=FC,alpha=0.87,label='MS adv.',zorder=3)
r2=axes[1].bar(x2+w2/2,adv_td,w2,color=GC, alpha=0.87,label='TD adv.',zorder=3)
axes[1].axhline(0,color='#333',lw=0.7,zorder=4)
# Value labels well above/below bars — no overlap with legend
for bar,col in [(r1,FC),(r2,GC)]:
    for b in bar:
        h=b.get_height()
        if abs(h)>0.05:
            ypos = h+0.10 if h>=0 else h-0.28
            axes[1].text(b.get_x()+b.get_width()/2, ypos,
                        f'{h:+.1f}',ha='center',va='bottom',
                        fontsize=5.8,color=col,fontweight='bold')
axes[1].set_xticks(x2); axes[1].set_xticklabels(fam_lbs,fontsize=7.5)
axes[1].set_ylabel('Adv. (MLP$-$FT-RLS)',fontsize=7,labelpad=3)
axes[1].set_title('Per-family adv.',fontsize=7.5,fontweight='bold',pad=3)
axes[1].yaxis.grid(True,ls=':',lw=0.35,alpha=0.7,zorder=0); axes[1].set_axisbelow(True)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
# Legend at top-left — values are in bottom half so no overlap
axes[1].legend(fontsize=6.5,loc='upper left',handlelength=1.0,handletextpad=0.4,
               frameon=True,edgecolor='#ccc',borderpad=0.4,labelspacing=0.3)
plt.savefig('/home/claude/paper/figures/fig6_winsummary.pdf',bbox_inches='tight'); plt.close(); print('Fig6 FIXED')
print('All done.')
