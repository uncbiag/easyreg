import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
# ax = sns.boxplot(x=tips["total_bill"])
# plt.show()
ax = sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()
# ax = sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="Set3")
# plt.show()