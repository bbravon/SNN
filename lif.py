import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class lifer:
    def __init__(self, inputs, U0, Uth, B):
        self.I = sum(inputs)
        self.U0 = U0
        self.Uth = Uth
        self.B = B
        self.U = np.zeros(len(self.I))
        self.U[0] = U0
        self.spikes = np.zeros(len(self.I))
    def update(self):
        # ver la edo de la membrana
        for t in range(len(self.I)-1):
            U_th=self.Uth*(self.U[t]>self.Uth)
            self.U[t+1] = self.B*self.U[t]+self.I[t]-U_th

        # generar el output 1 tiempo después de superar el umbral
        up=3*(self.U>=self.Uth)
        up2 = up[:-1] 
        shifted_data_padded = np.insert(up2, 0, 0) # Alternatively: np.concatenate(([0], data_slice))
        self.spikes = self.spikes + shifted_data_padded
  
    def plot(self,val=0):
        fig, axs = plt.subplots(3)
        axs[0].plot(self.I)
        axs[0].set_title("input")
        axs[1].plot(self.U)
        axs[1].set_title("membrana")
        axs[2].plot(self.spikes)
        axs[2].set_title("output")
        fig.tight_layout()

