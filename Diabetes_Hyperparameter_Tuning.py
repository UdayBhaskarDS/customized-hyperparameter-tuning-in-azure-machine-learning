
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required#
from azureml.core import Workspace, Dataset,Experiment,Environment
from azureml.core.run import Run


import numpy as np
import matplotlib as pyplot
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

subscription_id = 'ad4d14ed-5af6-4288-9a19-a81cdcaf5b42'
resource_group = 'aml_RG'
workspace_name = 'aml_ws'
run = Run.get_context()
workspace = Workspace(subscription_id, resource_group, workspace_name)
env = Environment.get(workspace=workspace, name="AzureML-Tutorial")

experiment = Experiment(workspace=workspace, name='Hyperparameter-Tuning_BestModel')

data = Dataset.get_by_name(workspace, name='Diabetes')
data = data.to_pandas_dataframe()
run.log(name="message",
            value= f"{type(data)}")
    
data_untouched = data.copy()

encode_ref={'Class(Outcome)':{'tested_positive':0,'tested_negative':1}} # Manually Encoding

data=data.replace(encode_ref)
data.head()

data['Class(Outcome)']=data['Class(Outcome)'].astype('category')

print(data.dtypes)

data['Class(Outcome)']=data['Class(Outcome)'].cat.codes  # Label encoding technique. Only two values available,so no need of creating dummy variables.

data = data.rename(columns={'Daibetes pedigree Function':'DPF'})
print(data.columns)

Q1=data['Pregnancies'].quantile(0.25)
Q3=data['Pregnancies'].quantile(0.75)
IQR=Q3-Q1

Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR

df = data[data['Pregnancies']< Upper_Whisker]
from scipy import stats

z=np.abs(stats.zscore(data['Pregnancies']))

## Check the points greater than 3, gives all variables in the dataset.
threshold=3

## Check the points lesser than 3, gives all variables in the dataset.
df1=data[(z< 3)]

Q1=data['Plasma Glucose'].quantile(0.25)
Q3=data['Plasma Glucose'].quantile(0.75)
IQR=Q3-Q1

Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR


df3 = data[data['Plasma Glucose']> Lower_Whisker]
df3.shape

z=np.abs(stats.zscore(data['Plasma Glucose']))


## Check the points greater than 3, gives all variables in the dataset.
threshold=3

## Check the points lesser than 3, gives all variables in the dataset.
df4=data[(z< 3)]

Q1=data['Bloodpressure'].quantile(0.25)
Q3=data['Bloodpressure'].quantile(0.75)
IQR=Q3-Q1

Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR

df15 = data[(data['Bloodpressure']> Upper_Whisker) & (data['Bloodpressure']<Lower_Whisker)]  # Removing outliers on both sides.
df15.shape

z=np.abs(stats.zscore(data['Bloodpressure']))

## Check the points greater than 3, gives all variables in the dataset.
threshold=3

## Check the points lesser than 3, gives all variables in the dataset.
df5=data[(z< 3)]

Q1=data['SkinThickness'].quantile(0.25)
Q3=data['SkinThickness'].quantile(0.75)
IQR=Q3-Q1

Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR

df6 = data[data['SkinThickness']> Upper_Whisker]


z=np.abs(stats.zscore(data['SkinThickness']))

## Check the points greater than 3, gives all variables in the dataset.
threshold=3

## Check the points lesser than 3, gives all variables in the dataset.
df7=data[(z< 3)]

Q1=data['Insulin'].quantile(0.25)
Q3=data['Insulin'].quantile(0.75)
IQR=Q3-Q1

Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)

df8 = data[data['Insulin']> Upper_Whisker]
df8.shape

z=np.abs(stats.zscore(data['Insulin']))

## Check the points greater than 3, gives all variables in the dataset.
threshold=3

## Check the points lesser than 3, gives all variables in the dataset.
df9=data[(z< 3)]

Q1=data['BMI'].quantile(0.25)
Q3=data['BMI'].quantile(0.75)
IQR=Q3-Q1
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR

df16 = data[(data['BMI']>Upper_Whisker) & (data['BMI']<Lower_Whisker)]  # Removing outliers on both sides.

z=np.abs(stats.zscore(data['BMI']))

## Check the points lesser than 3, gives all variables in the dataset.
df10=data[(z< 3)]

Q1=data['DPF'].quantile(0.25)
Q3=data['DPF'].quantile(0.75)
IQR=Q3-Q1

Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)

df11 = data[data['DPF']> Upper_Whisker]
df11.shape

z=np.abs(stats.zscore(data['DPF']))
print(z)

## Check the points greater than 3, gives all variables in the dataset.
threshold=3
print(np.where(z>3))

## Check the points lesser than 3, gives all variables in the dataset.
df12=data[(z< 3)]
print(df12)

Q1=data['Age'].quantile(0.25)
Q3=data['Age'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)

df13 = data[data['Age']> Upper_Whisker]
df13.shape

z=np.abs(stats.zscore(data['Age']))
print(z)

## Check the points greater than 3, gives all variables in the dataset.
threshold=3
print(np.where(z>3))

## Check the points lesser than 3, gives all variables in the dataset.
df14=data[(z< 3)]
print(df14)

##CAPPING
print(data['Pregnancies'].quantile(0.10))
print(data['Pregnancies'].quantile(0.90))

data['Pregnancies']=np.where(data['Pregnancies']<0.0,0.0,data['Pregnancies'])
data['Pregnancies']=np.where(data['Pregnancies']>9.0,9.0,data['Pregnancies'])


print(data['Plasma Glucose'].quantile(0.10))
print(data['Plasma Glucose'].quantile(0.90))

data['Plasma Glucose']=np.where(data['Plasma Glucose']<85.0,85.0,data['Plasma Glucose'])
data['Plasma Glucose']=np.where(data['Plasma Glucose']>167.0,167.0,data['Plasma Glucose'])

print(data['Bloodpressure'].quantile(0.10))
print(data['Bloodpressure'].quantile(0.90))

data['Bloodpressure']=np.where(data['Bloodpressure']<54.0,54.0,data['Bloodpressure'])
data['Bloodpressure']=np.where(data['Bloodpressure']>88.0,88.0,data['Bloodpressure'])

print(data['SkinThickness'].quantile(0.10))
print(data['SkinThickness'].quantile(0.90))

data['SkinThickness']=np.where(data['SkinThickness']<0.0,0.0,data['SkinThickness'])
data['SkinThickness']=np.where(data['SkinThickness']>40.0,40.0,data['SkinThickness'])   

data['Insulin']=np.where(data['Insulin']<0.0,0.0,data['Insulin'])
data['Insulin']=np.where(data['Insulin']>210.0,210.0,data['Insulin'])

data['BMI']=np.where(data['BMI']<23.6,23.6,data['BMI'])
data['BMI']=np.where(data['BMI']>41.5,41.5,data['BMI'])

data['DPF']=np.where(data['DPF']<0.165,0.165,data['DPF'])
data['DPF']=np.where(data['DPF']>0.8786,0.8786,data['DPF'])

data['Age']=np.where(data['Age']<22.0,22.0,data['Age'])
data['Age']=np.where(data['Age']>51.0,51.0,data['Age'])
run.log(name="message",
            value="training Started!!!")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,ConfusionMatrixDisplay,roc_auc_score,roc_curve,plot_roc_curve
#scikit learn
x=data.iloc[:,1:8]
y=data.iloc[:,8:]
    
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 111) 
    

C_param_range = [0.001,0.01,0.1,1,10,100]

performence_table = pd.DataFrame(columns = ['C_parameter','roc_auc_score'])
performence_table['C_parameter'] = C_param_range
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

# Root directory or working directory
import os
root_dir_path = os.getcwd()

# Create outputs folder
output_folder_path = "./outputs/"
reports_path = "./Performence_Report/"

os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(reports_path, exist_ok=True)

stepscores_dict = {}
j = 0
for i in C_param_range:
   
    
    # Apply logistic regression model to training data
    lr = LogisticRegression(penalty = 'l2', C = i,random_state = 0)
    lr.fit(x_train,y_train)
    model_name = "model_C_" + str(i) + ".pkl"
    filename = "outputs/" + model_name
    
    joblib.dump(value=lr, filename=filename)
    # Predict using model
    y_pred = lr.predict(x_test)
    
    # Saving accuracy score in table
    auc = roc_auc_score(y_test,y_pred)
    stepscores_dict[model_name] = auc
    
    
    run.log('roc_auc_score', auc)
    performence_table.iloc[j,1] = auc
    
    run.upload_file(name=model_name, path_or_stream=filename)
    print(model_name+'-'+str(roc_auc_score))

    
    j += 1

BestModel= max(stepscores_dict, key= lambda x: stepscores_dict[x])
print(BestModel)  

best_model_Loaded = joblib.load("outputs/"+BestModel)
os.makedirs("outputs/Modelfile_for_OP", exist_ok=True)
joblib.dump(value=best_model_Loaded, filename="outputs/"+"BestModel.pkl")
joblib.dump(value=best_model_Loaded, filename="outputs/Modelfile_for_OP/"+"BestModel.pkl")

    

saved_path = reports_path + "performence_table.pkl"
f = open(saved_path, 'wb')
joblib.dump(performence_table, f)
f.close()



maximum_score_runid = None
maximum_score = None

for run in experiment.get_runs():
    run_metrics = run.get_metrics()
    print('metrics:',run_metrics)
    run_details = run.get_details()
    # each logged metric becomes a key in this returned dict
    run_score = run_metrics["roc_auc_score"]
    run = run_details["runId"]
    print('auc_score', run_score)
    print('run_step', run)
from azureml.core.model import Model
  
model = Model.register(workspace =workspace ,model_name='Diabetes_Classification1', model_path="outputs/Modelfile_for_OP/BestModel.pkl",)
print(model.name, model.id, model.version, sep='\t')
"""print("Best run_step: " + maximum_score_runid)
print("Best run_step_Score: " + str(run_score))

from azureml.core import Run
best_run = Run(experiment=experiment, run_id=maximum_score_runid)
print(best_run.get_file_names())

#best_run.download_file(name="model_C.pkl")"""




