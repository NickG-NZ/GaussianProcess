import numpy as np
import matplotlib.pyplot as plt


# Visualize the output of a 1D gaussian process
xTrain = np.array([-0.686642, -0.198111, -0.740419, -0.782382, 0.997849])
yTrain = np.array([-0.563486, 0.0258648, 0.678224, 0.22528, -0.407937])

xTest = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

# Predictions for different noise levels
# sigma = 0.001
yPred1 = np.array([1.09738, 0.668775, 0.309657, 0.0340701, -0.149947, -0.241002, -0.24444, -0.171891,
                 -0.0402797, 0.129578, 0.314699, 0.491795, 0.639112, 0.738089, 0.774685, 0.740278,
                  0.632077, 0.453039, 0.211322, -0.0806386, -0.407323])

# sigma = 0
yPred2 = np.array([-34.3715, -10.5731, -0.424533, -0.0974322, -5.19407, -11.2615, -14.3057, -11.2468,
                    -0.262643, 19.0169, 45.4864, 76.7112, 109.223, 138.931, 161.586, 173.266, 170.804,
                    152.126, 116.47, 64.459, -1.96895])

fig, ax = plt.subplots(2, 1)
ax[0].scatter(xTrain, yTrain, marker='+')
ax[0].plot(xTest, yPred1)
ax[0].set(xlabel="x-value", ylabel="y-value", title="Gaussian Process (sigma = 0.001)")
ax[0].grid()

ax[1].scatter(xTrain, yTrain, marker='+')
ax[1].plot(xTest, yPred2)
ax[1].set(xlabel="x-value", ylabel="y-value", title="Gaussian Process (sigma = 0)")
ax[1].grid()

plt.show()
