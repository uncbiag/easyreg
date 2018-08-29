import matplotlib.pyplot as plt

from model_pool.utils import sigmoid_decay, sigmoid_explode

x = list(range(50))
y = [ sigmoid_decay(iterm,static=5, k=4)*10 for iterm in x]
plt.plot(x,y)
plt.xlabel('epoch')
plt.title('inverse sigmoid decay factor')
plt.show()



y = [min(sigmoid_explode(iterm,static=5, k=4)*0.005,0.5) for iterm in x]
plt.plot(x,y)
plt.xlabel('epoch')
plt.title('inverse sigmoid decay factor')
plt.show()