import seaborn as sns
sns.pairplot(train[cont_columns], vars=['cont1','cont2','cont3','cont4','loss'], kind = 'scatter',diag_kind='kde')
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.figure(figsize=(15,25))
gs = gridspec.GridSpec(7, 2)

for i, cn in enumerate(train[cont_columns].columns):
    ax = plt.subplot(gs[i])
    stats.probplot(train[cn], dist = stats.norm, plot = ax)
    ax.set_xlabel('')
    ax.set_title('Probplot of feature: cont' + str(i+1))
	
