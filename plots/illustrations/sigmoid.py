import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
sigmoid = 1./(1+np.exp(-x))
step = x>0.0


plt.plot(x, sigmoid)
plt.xlabel("x", size=15)
plt.ylabel("Sigmoid function", size=15)
plt.tight_layout()
plt.savefig("sigmoid.png")
plt.clf()

plt.plot(x, step)
plt.xlabel("x", size=15)
plt.ylabel("Step function", size=15)
plt.tight_layout()
plt.savefig("step.png")
plt.clf()

plt.plot(x, step)
plt.plot(x, sigmoid)
plt.xlabel("x", size=15)
plt.ylabel("Activation function", size=15)
plt.legend(["Step function","Sigmoid function"])
plt.tight_layout()
plt.savefig("step_sigmoid.png")
plt.clf()