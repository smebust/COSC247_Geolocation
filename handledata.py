import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def splitPstData(data):

    lats = np.array([data[i,4] for i in range(data.shape[0])])
    longs = np.array([data[i,5] for i in range(data.shape[0])])
    ys = np.transpose(np.array([lats, longs]))
    #print(ys)

    Xs = []
    for i in range(data.shape[1]):
        if (i != 4) and (i != 5):
            thisRow = np.array([data[j,i] for j in range(data.shape[0])])
            Xs.append(thisRow)

    Xs = np.transpose(np.array(Xs))
    #print(Xs)

    return (Xs, ys)

def ignoreNullIsland(data):

    nullless = []
    for i in range(data.shape[0]):
        if (data[i,7] >= -1000):
            if not((data[i,4] == 0) and (data[i,5] == 0)):
                nullless.append(data[i,:])

    nullless = np.array(nullless)
    return (nullless)

def joinFriends_train(friends, posts):
    friendAvs = np.loadtxt(str(friends), delimiter = ',')
    postsData = np.loadtxt(str(posts), delimiter=',', skiprows=1)

    toRetList = np.zeros((postsData.shape[0], 13))
    for i in range(postsData.shape[0]):
        newRow = np.append(postsData[i,:], friendAvs[i,:])

        toRetList[i,:] = newRow

    return np.array(toRetList)

def joinFriends_test(friends, posts):
    friendAvs = np.loadtxt(str(friends), delimiter = ',')
    postsData = np.loadtxt(str(posts), delimiter=',', skiprows=1)

    toRetList = np.zeros((postsData.shape[0], 11))
    for i in range(postsData.shape[0]):
        newRow = np.append(postsData[i,:], friendAvs[i,:])

        toRetList[i,:] = newRow

    return np.array(toRetList)

def friends_test(edges, train, test):
    
    loc = np.zeros((test.shape[0], 6))
    for i in range(test.shape[0]):
        hour1 = []
        hour2 = []
        hour3 = []
        lat = []
        lon = []
        ps = []
        for j in range(edges.shape[0]):
            if test[i,0] == edges[j,0]:
                for k in range(train.shape[0]):
                    if train[k,0] == edges[j,1]:
                        hour1.append(train[k,1])
                        hour2.append(train[k,2])
                        hour3.append(train[k,3])
                        lat.append(train[k,4])
                        lon.append(train[k,5])
                        ps.append(train[k,6])
                for k in range(test.shape[0]):
                    if test[k,0] == edges[j,1]:
                        hour1.append(test[k,1])
                        hour2.append(test[k,2])
                        hour3.append(test[k,3])
                        ps.append(test[k,4])
        loc[i,0] = 0                  
        loc[i,1] = 0                 
        loc[i,2] = 0
        loc[i,3] = 0
        loc[i,4] = 0
        loc[i,5] = 0
        if len(hour1) > 0:
            loc[i,0] = np.mean(hour1)                    
        if len(hour2) > 0:
            loc[i,1] = np.mean(hour2)     
        if len(hour2) > 0:
            loc[i,2] = np.mean(hour3)
        if len(lat) > 0:  
            loc[i,3] = np.mean(lat)
        if len(lon) > 0:
            loc[i,4] = np.mean(lon)
        if len(ps) > 0:
            loc[i,5] = np.mean(ps)
        
        print()
        print(i)
        print()
        
    return(loc)

def friends_train(edges, posts):
    
    loc = np.zeros((posts.shape[0], 6))
    for i in range(posts.shape[0]):
        hour1 = []
        hour2 = []
        hour3 = []
        lat = []
        lon = []
        ps = []
        for j in range(edges.shape[0]):
            if posts[i,0] == edges[j,0]:
                for k in range(posts.shape[0]):
                    if posts[k,0] == edges[j,1]:
                        if not((posts[k,4] == 0) and (posts[k,5] == 0)):
                            hour1.append(posts[k,1])
                            hour2.append(posts[k,2])
                            hour3.append(posts[k,3])
                            lat.append(posts[k,4])
                            lon.append(posts[k,5])
                            ps.append(posts[k,6])
        loc[i,0] = np.mean(hour1)                    
        loc[i,1] = np.mean(hour2)                    
        loc[i,2] = np.mean(hour3)      
        loc[i,3] = np.mean(lat)
        loc[i,4] = np.mean(lon)
        loc[i,5] = np.mexan(ps)
        
        print()
        print(i)
        print()
        
    return(loc)

if __name__ == "__main__":
    pstrn = np.loadtxt("posts_train.txt", delimiter = ',', skiprows=1)
    #print(pstrn)
    #print(pstrn.shape)
    pstst = np.loadtxt("posts_test.txt", delimiter = ',', skiprows=1)
    #print(pstst)
    #print(pstst.shape)
    edges = np.loadtxt("graph.txt")
    #print(edges)
    #print(edges.shape)
    splitPstData(pstrn)

    ftest = friends_test(edges, pstrn, pstst)
    np.savetxt("friends_test.txt", ftest, delimiter=",")

    ftrain = friends_train(edges, pstrn, pstst)
    np.savetxt("friends_train.txt", ftrain, delimiter=",")
    
    full_train = joinFriends_train("friends.txt", "posts_train.txt")
    print(full_train)
    np.savetxt("full_train.txt", full_train, delimiter=",")
 
    full_test = joinFriends_test("friends_test.txt", "posts_test.txt")
    print(full_test)
    np.savetxt("full_test.txt", full_test, delimiter=",")