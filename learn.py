import numpy as np
import sklearn
import handledata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    pstrn = np.loadtxt("full_train.txt", delimiter = ',', skiprows=0)
    #print(pstrn)
    #print(pstrn.shape)
    pstst = np.loadtxt("full_test.txt", delimiter = ',', skiprows=0)
    #print(pstst)
    #print(pstst.shape)
    
    pstrn = handledata.ignoreNullIsland(pstrn)
    
    X, y = handledata.splitPstData(pstrn)
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25)
    
    reg = LinearRegression(normalize=True)
    reg.fit(X_tr, y_tr)
    preds_tr = reg.predict(X_tr)
    
    print(np.sqrt(mean_squared_error(y_tr, preds_tr)))
    
    preds = reg.predict(pstst)
    submission = np.zeros((preds.shape[0], 3))    
    for i in range(submission.shape[0]):
        submission[i,0] = int(pstst[i,0])
        submission[i,1] = preds[i,0]
        submission[i,2] = preds[i,1]
    print(submission)
    np.savetxt("submission_example.txt", submission, delimiter=',', fmt=['%d','%f','%f'], header="Id,Lat,Lon",comments='',)
