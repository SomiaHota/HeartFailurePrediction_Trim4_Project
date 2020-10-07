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
# Data preprocessing
dataset.isnull().sum()

# Correlation
data = dataset.iloc[:, 1:-1]
corr = data.corr()
corr.head()
sns.heatmap(corr)
# Feature Selection

plt.rcParams['figure.figsize']=15,6 
sns.set_style("darkgrid")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()
# Box Plots
sns.boxplot(x = dataset.ejection_fraction, color = 'green')
plt.show()
dataset[dataset['ejection_fraction']>=70]
dataset = dataset[dataset['ejection_fraction']<70]
sns.boxplot(x=dataset.time, color = 'green')
plt.show()
sns.boxplot(x=dataset.serum_creatinine, color = 'green')
plt.show()

# Data Visualization using Plotly
# Age vs Count
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(
    x = dataset['age'],
    xbins=dict( # bins used for histogram
        start=40,
        end=95,
        size=2
    ),
    marker_color='#e8ab60',
    opacity=1
))

fig.update_layout(
    title_text='AGE DISTRIBUTION',
    xaxis_title_text='AGE',
    yaxis_title_text='COUNT', 
    bargap=0.05, # gap between bars of adjacent location coordinates
    xaxis =  {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)

fig.show()
# Age vs Death Event
import plotly.express as px
fig = px.histogram(dataset, x="age", color="DEATH_EVENT", hover_data=dataset.columns, 
                   title ="Distribution of AGE Vs DEATH_EVENT", 
                   labels={"age": "AGE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()
# CREATININE PHOSPHOKINASE vs Count
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(
    x = dataset['creatinine_phosphokinase'],
    xbins=dict( # bins used for histogram
        start=23,
        end=582,
        size=15
    ),
    marker_color='#FE6F5E',
    opacity=1
))

fig.update_layout(
    title_text='CREATININE PHOSPHOKINASE DISTRIBUTION',
    xaxis_title_text='CREATININE PHOSPHOKINASE',
    yaxis_title_text='COUNT', 
    bargap=0.05, # gap between bars of adjacent location coordinates
    xaxis =  {'showgrid': False },
    yaxis = {'showgrid': False },
    template = 'plotly_dark'
)

fig.show()

# CREATININE PHOSPHOKINASE vs Death Event
import plotly.express as px
fig = px.histogram(dataset, x="creatinine_phosphokinase", color="DEATH_EVENT", hover_data=dataset.columns,
                   title ="Distribution of CREATININE PHOSPHOKINASE Vs DEATH_EVENT", 
                   labels={"creatinine_phosphokinase": "CREATININE PHOSPHOKINASE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()

# EJECTION FRACTION Vs DEATH_EVENT
import plotly.express as px
fig = px.histogram(dataset, x="ejection_fraction", color="DEATH_EVENT", hover_data=dataset.columns,
                   title ="Distribution of EJECTION FRACTION Vs DEATH_EVENT", 
                   labels={"ejection_fraction": "EJECTION FRACTION"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"})
fig.show()

# Gender Vs DEATH_EVENT
import plotly.graph_objects as go
from plotly.subplots import make_subplots

d1 = dataset[(dataset["DEATH_EVENT"]==0) & (dataset["sex"]==1)]
d2 = dataset[(dataset["DEATH_EVENT"]==1) & (dataset["sex"]==1)]
d3 = dataset[(dataset["DEATH_EVENT"]==0) & (dataset["sex"]==0)]
d4 = dataset[(dataset["DEATH_EVENT"]==1) & (dataset["sex"]==0)]

label1 = ["Male","Female"]
label2 = ['Male - Survived','Male - Died', "Female -  Survived", "Female - Died"]
values1 = [(len(d1)+len(d2)), (len(d3)+len(d4))]
values2 = [len(d1),len(d2),len(d3),len(d4)]

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=label1, values=values1, name="GENDER"),
              1, 1)
fig.add_trace(go.Pie(labels=label2, values=values2, name="GENDER VS DEATH_EVENT"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent")


fig.update_layout(
    title_text="GENDER DISTRIBUTION IN THE DATASET  \
                   GENDER VS DEATH_EVENT",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='GENDER', x=0.19, y=0.5, font_size=10, showarrow=False),
                 dict(text='GENDER VS DEATH_EVENT', x=0.84, y=0.5, font_size=9, showarrow=False)],
    autosize=False,width=1200, height=500, paper_bgcolor="white")

fig.show()


# Gender Vs Smoking
from plotly.subplots import make_subplots

d1 = dataset[(dataset["smoking"]==0) & (dataset["sex"]==1)]
d2 = dataset[(dataset["smoking"]==1) & (dataset["sex"]==1)]
d3 = dataset[(dataset["smoking"]==0) & (dataset["sex"]==0)]
d4 = dataset[(dataset["smoking"]==1) & (dataset["sex"]==0)]

label1 = ['Male - Nonsmoker','Male - Smoker', "Female -  Nonsmoker", "Female - Smoker"]
values1 = [len(d1),len(d2),len(d3),len(d4)]

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
fig.add_trace(go.Pie(labels=label1, values=values1, name="GENDER VS SMOKING"),1,1)
              

fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="GENDER VS SMOKING",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='GENDER VS SMOKING', x=0.50, y=0.5, font_size=9, showarrow=False)],
    autosize=False,width=1200, height=500, paper_bgcolor="white")

fig.show()


# Gender Vs Anaemia
from plotly.subplots import make_subplots

d1 = dataset[(dataset["anaemia"]==0) & (dataset["sex"]==1)]
d2 = dataset[(dataset["anaemia"]==1) & (dataset["sex"]==1)]
d3 = dataset[(dataset["anaemia"]==0) & (dataset["sex"]==0)]
d4 = dataset[(dataset["anaemia"]==1) & (dataset["sex"]==0)]

label2 = ['Male - Nonanaemic','Male - Anaemic', "Female -  Nonanaemic", "Female - Anaemic"]
values2 = [len(d1),len(d2),len(d3),len(d4)]

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=1, specs=[[{'type':'domain'}]])
fig.add_trace(go.Pie(labels=label2, values=values2, name="GENDER VS ANAEMIA"),
              1, 1)

fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="                   GENDER VS ANAEMIA",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='GENDER VS ANAEMIA', x=0.50, y=0.5, font_size=9, showarrow=False)],
    autosize=False,width=1200, height=500, paper_bgcolor="white")

fig.show()


# High BP Vs Smoking
from plotly.subplots import make_subplots

d1 = dataset[(dataset["high_blood_pressure"]==0) & (dataset["smoking"]==1)]
d2 = dataset[(dataset["high_blood_pressure"]==1) & (dataset["smoking"]==1)]
d3 = dataset[(dataset["high_blood_pressure"]==0) & (dataset["smoking"]==0)]
d4 = dataset[(dataset["high_blood_pressure"]==1) & (dataset["smoking"]==0)]

label1 = ["Smoker","Non-Smoker"]
label2 = ['Smoker - No High BP','Smoker - With High BP', "NonSmoker - No High BP", "NonSmoker - With High BP"]
values1 = [(len(d1)+len(d2)), (len(d3)+len(d4))]
values2 = [len(d1),len(d2),len(d3),len(d4)]

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=label1, values=values1, name="SMOKER"),
              1, 1)
fig.add_trace(go.Pie(labels=label2, values=values2, name="SMOKER VS HIGH BP"),
              1, 2)

fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="SMOKER DISTRIBUTION IN THE DATASET  \
                   SMOKER VS HIGH BP",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='SMOKER', x=0.19, y=0.5, font_size=10, showarrow=False),
                 dict(text='SMOKER VS HIGH BP', x=0.84, y=0.5, font_size=9, showarrow=False)],
    autosize=False,width=1200, height=500, paper_bgcolor="white")

fig.show()


# Logistic Regression using entire Feature set for prediction of Death_Event
x = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
mylist1 = []
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist1.append(ac)
print(cm)
print(ac)

#  KNeighborsClassifier using entire Feature set for prediction of Death_Event
x = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
list1 = []
for neighbors in range(2,10):
    classifier = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    list1.append(accuracy_score(y_test,y_pred))
plt.plot(list(range(2,10)), list1)
plt.show()
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist1.append(ac)
print(cm)
print(ac)

#  RandomForestClassifier using entire Feature set for prediction of Death_Event
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
list1 = []
for estimators in range(1,30):
    classifier = RandomForestClassifier(n_estimators = estimators, random_state=0, criterion='entropy')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    list1.append(accuracy_score(y_test,y_pred))
#print(mylist)
plt.plot(list(range(1,30)), list1)
plt.show()
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 11, criterion='entropy', random_state=0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
# Making the confusion matrix and calculating the accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist1.append(ac)
print(cm)
print(ac)

#  Plotting of Accuracy results of "Logistic Regression", "KNearestNeighbours","RandomForest" Classifiers for entire Feature set
mylistnew = ["Logistic Regression", "KNearestNeighbours","RandomForest"]
plt.rcParams['figure.figsize']=15,6 
sns.set_style("darkgrid")
ax = sns.barplot(x=mylistnew, y=mylist1, palette = "hls", saturation =1.5)
plt.xlabel("Classifier Models", fontsize = 20 )
plt.ylabel("% of Accuracy", fontsize = 20)
plt.title("Accuracy of different Classifier Models", fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


#  LOGISTIC REGRESSION with the 3 important features found out using ExtraTreesClassifier
x = dataset.iloc[:, [4,7,11]]
y = dataset.iloc[:,-1]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
mylist = []
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist.append(ac)
print(cm)
print(ac)

#  KNeighborsClassifier with the 3 important features found out using ExtraTreesClassifier
x = dataset.iloc[:, [4,7,11]]
y = dataset.iloc[:,-1]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
list1 = []
for neighbors in range(3,10):
    classifier = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    list1.append(accuracy_score(y_test,y_pred))
plt.plot(list(range(3,10)), list1)
plt.show()
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist.append(ac)
print(cm)
print(ac)

#  RandomForestClassifier with the 3 important features found out using ExtraTreesClassifier
x = dataset.iloc[:, [4,7,11]]
y = dataset.iloc[:,-1]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
list1 = []
for estimators in range(5,30):
    classifier = RandomForestClassifier(n_estimators = estimators, random_state=0, criterion='entropy')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    list1.append(accuracy_score(y_test,y_pred))
#print(mylist)
plt.plot(list(range(5,30)), list1)
plt.show()
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion='entropy', random_state=0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
# Making the confusion matrix and calculating the accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
mylist.append(ac)
print(cm)
print(ac)

#  Plotting of Accuracy results of "Logistic Regression", "KNearestNeighbours","RandomForest" Classifiers for the 3 important features found out using ExtraTreesClassifier
mylist2 = ["Logistic Regression", "KNearestNeighbours","RandomForest"]
plt.rcParams['figure.figsize']=15,6 
sns.set_style("darkgrid")
ax = sns.barplot(x=mylist2, y=mylist, palette = "rocket", saturation =1.5)
plt.xlabel("Classifier Models", fontsize = 20 )
plt.ylabel("% of Accuracy", fontsize = 20)
plt.title("Accuracy of different Classifier Models", fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()
