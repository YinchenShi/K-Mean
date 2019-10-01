import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##load data from csv file
def load_csv():
    dataset = []
    f = open("./dataset_noclass.csv",'r')
    for lines in f:
        dataset.append(lines)
    for i in range(len(dataset)):
        dataset[i] = dataset[i].split(',')
        dataset[i][-1] = dataset[i][-1][0:-1]
    del dataset[0]
    for k in range(len(dataset)):
        for j in range(len(dataset[k])):
            dataset[k][j] = float(dataset[k][j])
    return dataset

dataset = load_csv()

##calculate the euclidean distance of 3 dimensional data sets
def Euclidean(p1,p2):
    distance = 0
    for i in range(len(p1)):
        distance += (p1[i] - p2[i])**2
    distance = math.sqrt(distance)
    return distance


##we randomize the centroids
def Randomization(num):
    weights = []
    for i in range(num):
        a = random.randint(-1299,2325)/1000
        ## range of x from -1.29904731 to  2.325876798
        b = random.randint(-878,1453)/1000
        ## range of y from -0.878917218 to 1.453850114
        c = random.randint(-1006,2010)/1000
        ## range of y from -1.006866904 to 2.01089859
        weights.append([a,b,c])
    return weights


def train_network(Centroids,dataset): #1st parameter is randomly selected initial centroids
                                          #2nd parameter is the dataset
    cluster1 = []
    #create first cluster list
    cluster2 = []
    #create first cluster list
    x=0
    y=0
    z = 0
    ##x, y, z coordinates sum respectively
    new_weight_1=[]
    new_weight_2 = []
    Euclidean_Dis = 0
    #current euclidean 
    Euclidean_Pre = 0
    #previous euclidean 
    while Euclidean_Dis <= Euclidean_Pre:
        ##if we have current data not better than previous
        ## then we stop training
        Euclidean_Pre = Euclidean_Dis
        for i in range(len(dataset)):
            if Euclidean(Centroids[0],dataset[i]) < Euclidean(Centroids[1],dataset[i]):
                cluster1.append(dataset[i])
            else:
                cluster2.append(dataset[i])
        for i in range(len(cluster1)):
            x += cluster1[i][0]
            y += cluster1[i][1]
            z += cluster1[i][2]
            ##take the total sum divide the length in cluster 1
        new_weight_1 = [x/len(cluster1),y/len(cluster1),z/len(cluster1)]
        x,y,z = 0,0,0
        for j in range(len(cluster2)):
            x += cluster2[j][0]
            y += cluster2[j][1]
            z += cluster2[j][2]
            ##take the total sum divide the length in cluster 2
        new_weight_2 = [x/len(cluster2),y/len(cluster2),z/len(cluster2)]

        for k in range(len(cluster1)):
            Euclidean_Dis += math.sqrt(Euclidean(cluster1[k],new_weight_1))
        for l in range(len(cluster2)):
            Euclidean_Dis += math.sqrt(Euclidean(cluster2[l],new_weight_2))
    return [new_weight_1,new_weight_2]

init = Randomization(2)
final_result = train_network(init,dataset)

def main(dataset):
    #draw the 3d diagram to show the cluster groupe
    x1 = dataset
    fig = plt.figure(figsize=(8, 8))
    d = fig.gca(projection='3d')
    for x in x1:
        d.scatter(x[0],x[1],x[2],c='b')
    d.scatter(final_result[0][0],final_result[0][1],final_result[0][2],c='r', marker = '*')
    d.scatter(final_result[1][0],final_result[1][1],final_result[1][2],c='r', marker = '*')
    plt.savefig("3D_Diagram.png")
    plt.show()

main(dataset)

#we write data classification points
def write(net):
    f = open("K-means.txt",'w')
    count1 =0
    count2 = 0
    for i in range(len(dataset)):
        if Euclidean(dataset[i], net[0]) <= Euclidean(dataset[i], net[1]):
            f.write("Type 1: Datapoint " + str(i+1) + "\n")
            count1 +=1
        else:
            f.write("Type 2: Datapoint " + str(i+1) + "\n")
            count2 +=1
    f.close()
    print(count1,"Type 1, datapoints in total")
    print(count2, "Type 2, datapoints in total")
    return
write(final_result)
print(final_result)
            
        



    
    



