# HeartFailurePrediction_Trim4_Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files 
  
  
uploaded = files.upload()
import io 
  
dataset = pd.read_csv(io.BytesIO(uploaded['heart_failure_clinical_records_dataset.csv'])) 
print(dataset)
dataset.head()
dataset.isnull().sum()
sns.boxplot(x = dataset.ejection_fraction, color = 'green')
plt.show()
dataset[dataset['ejection_fraction']>=70]
