import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from evaluate import y_true, y_pred

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Normal","Defect"])
plt.yticks([0,1], ["Normal","Defect"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png")
plt.show()

