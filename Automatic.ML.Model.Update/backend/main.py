from fastapi import FastAPI
import pandas as pd
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
from time import sleep
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import firebase_admin
import datetime
from firebase_admin import db



cred_obj = firebase_admin.credentials.Certificate('autotrainml-firebase-adminsdk-jfvyf-6849db5936.json')
default_app = firebase_admin.initialize_app(cred_obj, {
    	'databaseURL':"https://autotrainml-default-rtdb.asia-southeast1.firebasedatabase.app/"
    	})


retrain = False


def fireBaseDBInsertCurrentAccuracy(accuracy):
    
    
    # firebase_admin.initialize_app(cred_obj)
    
    ref = db.reference("/")
    model_acc = ref.child('ModelAcc')
    # stats = ref.child("ModelStats")
    model_acc.delete()
    # ct stores current time
    
    ct = str(datetime.datetime.now())
    
  	# dot cannot be in the keyname
    uniqueKey = ct.replace(".","")
    model_acc.update({
   	uniqueKey:{
              "TimeStamp":str(ct),
    			"accurracy":accuracy
    		}
    })


def fireBaseDBInsertHistory(modelTypes,accuracy):
    
    
    # firebase_admin.initialize_app(cred_obj)
    
    ref = db.reference("/")
    users_ref = ref.child('ModelHistory')
    # stats = ref.child("ModelStats")
    # users_ref.delete()
    # ct stores current time
    
    ct = str(datetime.datetime.now())
    
  	# dot cannot be in the keyname
    uniqueKey = ct.replace(".","")
    users_ref.update({
   	uniqueKey:{
              "TimeStamp":str(ct),
   			"modelType":modelTypes,
    			"accurracy":accuracy
    		}
    })
    
def fireBaseDBInsertCurrentModel(modelTypes,accuracy):
    
    
    # firebase_admin.initialize_app(cred_obj)
    
    ref = db.reference("/")
    # users_ref = ref.child('ModelHistory')
    stats = ref.child("CurrentModel")
    stats.delete()
    # ct stores current time
    
    ct = str(datetime.datetime.now())
    
  	# dot cannot be in the keyname
    uniqueKey = ct.replace(".","")
    stats.update({
   	uniqueKey:{
              # "TimeStamp":str(ct),
   			"modelType":modelTypes,
    			"accurracy":accuracy
    		}
    })
    
def fireBaseDBInsertModelStatus(modelStatus):
    
    
    # firebase_admin.initialize_app(cred_obj)
    
    ref = db.reference("/")
    # users_ref = ref.child('ModelHistory')
    status = ref.child("ModelStatus")
    status.delete()
    # ct stores current time
    
    ct = str(datetime.datetime.now())
    
  	# dot cannot be in the keyname
    uniqueKey = ct.replace(".","")
    status.update({
   	uniqueKey:{
              # "TimeStamp":str(ct),
   			"modelType":modelStatus
    		}
    })

def loadData(path):  #data loading
    dataFrame = pd.read_csv(path)
    return dataFrame
    
def data_wrangling(dataFrame):
    def fillMissingValue(dataFrame):
        dataFrame = dataFrame.fillna(method="ffill")
        return dataFrame

    df = fillMissingValue(dataFrame)
    return df
    

def getAccuracy(model, testX, testY):
    predictions = model.predict(testX)
    accuracy = accuracy_score(testY, predictions)*100
    print("Accuracy: ", accuracy)
    return accuracy


def training_algo(df):
    def RandomForest(trainX, trainY):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(trainX , trainY)
        return model

    def DecisionTree(trainX, trainY):
        model = sk.tree.DecisionTreeClassifier()
        model.fit(trainX , trainY)
        return model
    
    def KNN(trainX, trainY):
        model = sk.neighbors.KNeighborsClassifier()
        model.fit(trainX , trainY)
        return model
    
    def SupportVector(trainX, trainY):
        model = sk.svm.SVC(kernel="rbf")
        model.fit(trainX , trainY)
        return model
    fireBaseDBInsertModelStatus("Training")
    accuracyList = []
    X = df.values[:, 0:16]
    Y = df.values[:, 17]
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size = 0.25)
    model1 = RandomForest(X_train,Y_train)
    accuracyList.append(getAccuracy(model1, x_test, y_test))
    modelName = 'Random Forest'
    fireBaseDBInsertHistory(modelName,accuracyList[0])
    model2 = DecisionTree(X_train,Y_train)
    accuracyList.append(getAccuracy(model2, x_test, y_test))
    modelName = 'Decision Tree'
    fireBaseDBInsertHistory(modelName,accuracyList[1])
    model3 = KNN(X_train,Y_train)
    accuracyList.append(getAccuracy(model3, x_test, y_test))
    modelName = 'KNN'
    fireBaseDBInsertHistory(modelName,accuracyList[2])
    model4 = SupportVector(X_train,Y_train)
    accuracyList.append(getAccuracy(model4, x_test, y_test))
    modelName = 'SVM'
    fireBaseDBInsertHistory(modelName,accuracyList[3])
    
    accuracyMax = max(accuracyList)
    print("Max accuracy found " + str(accuracyMax))
    if (accuracyList.index(accuracyMax) == 0):
        print("Returning Model 1 -- Random Forest ")   
        fireBaseDBInsertCurrentModel("Random Forest",accuracyMax)
        return model1
    elif (accuracyList.index(accuracyMax) == 1):
        print("Returning Model 2 -- Decision Tree ")
        fireBaseDBInsertCurrentModel("Decision Tree",accuracyMax)
        return model2
    elif (accuracyList.index(accuracyMax) == 2):
        print("Returning Model 3 -- KNN ")
        fireBaseDBInsertCurrentModel("KNN",accuracyMax)
        return model3
    elif (accuracyList.index(accuracyMax) == 3):
        print("Returning Model 4 -- SVM ")
        fireBaseDBInsertCurrentModel("SVM",accuracyMax)
        return model4


def main(df = None):
    global retrain
    if(retrain == False):
        df = loadData("Dataset/online_shoppers_intention.csv")
        df = data_wrangling(df)
        train_df = df[0:int((len(df)/2))] 
        test_df  = df[(int(len(df)/2)):]
    if(retrain == True):
        train_df = df
        df = loadData("Dataset/online_shoppers_intention.csv")
        df = data_wrangling(df)
        test_df  = df[int(len(train_df)):]
        print("retraining")
    fireBaseDBInsertCurrentAccuracy(100.0)
    mod = training_algo(train_df)
    fireBaseDBInsertModelStatus("Deployed")
    count = 0 
    count_predicted = 0
    old_acc = 100
    for x in range(len(test_df)):
        # dd=pd.DataFrame(x)
        aa = df.loc[x]
        predictions = mod.predict(aa.values[0:16].reshape(1,-1))
        accuracy = accuracy_score(aa.values[17].reshape(1,-1), predictions)*100
        count += 1
        if(accuracy == 100):
            count_predicted +=1
        # else:
            # print("Accuracy: ", accuracy)
            # print("yay")
        accuracy = (count_predicted/count)*100
        if(accuracy < 99):
            retrain = True
            retrain_data = pd.concat([train_df, test_df[0:x]], axis=0)
            main(retrain_data)
        print("Accuracy: ", accuracy)
        if(old_acc != accuracy):
            fireBaseDBInsertCurrentAccuracy(accuracy)
    # predictions = mod.predict(test_df.values[:,0:16])
    # accuracy = accuracy_score(test_df.values[:,17], predictions)*100
    # print("Accuracy: ", accuracy)



origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/start/')
async def create_data_file():

    # print(pd.read_csv(StringIO(str(file.file.read(), 'utf-8')), encoding='utf-8'))
    ref = db.reference("/")
    users_ref = ref.child('ModelHistory')
    users_ref.delete()
    main()

    return {'No data'}









# def foo(bar, baz):
#     sleep(3)
#     print("Inside")
#     sleep(3)
#     print("After sleep")
#     sleep(3) 
#     print("After sleep2.0")
#     return ('foo' + baz)


# pool = ThreadPool(processes=1)



# async_result = pool.apply_async(foo, ('world', 'foo')) 



# return_val = async_result.get()

# print(return_val)