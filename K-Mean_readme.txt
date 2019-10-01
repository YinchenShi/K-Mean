K-Means
we have 3 inputs layers and 2 outputs layers3, this is because we have 3 dimensional data or 3 features set which is x, y, z respectively and each represents one node. Also two outputs neurons final weight is cluster centroids

Data preprocessing: I get rid of first line which is specification of 3 sets of data when i read csv file and add data to a list. Then i convert them to float so we could calculate it in propagation way right.

we implement euclidean distance to cluster the data into different groups
==========================================================================================
weights and total counts:
331 Type 1, datapoints in total
669 Type 2, datapoints in total
[[-0.3932096494676614, 0.06400010696019896, 0.2671741982985074], [0.7203479128623284, 0.3003386953954945, 0.5586304490312891]]