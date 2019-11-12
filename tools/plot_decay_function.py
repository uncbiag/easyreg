import matplotlib.pyplot as plt

from easyreg.utils import sigmoid_decay, sigmoid_explode

x = list(range(200))
y = [ sigmoid_decay(iterm,static=5, k=10)*10 for iterm in x]
color='brown'
plt.plot(x,y,color=color,linewidth=3.0)
plt.xlabel('epoch',fontsize=15)
plt.ylabel('value',fontsize=15)
plt.title('Inverse Sigmoid Decay Factor',fontsize=20)
plt.show()