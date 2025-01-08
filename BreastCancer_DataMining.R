#install packages and import dataset ####
library(cluster)
library(factoextra)
library(pastecs)
library(clustMixType)

library(tree)
library(randomForest)
library(pastecs)
library(caret)

cancer_data = read.csv("/Users/lauren/Desktop/DataMining/data.csv") 

#investigate data ####
head(cancer_data) #view the first few rows of the data
pastecs::stat.desc(cancer_data) #each row provides statistics (all numeric)
names(cancer_data)
summary(cancer_data) #notice X is all NA's, will remove X column and id column in pre-processing
str(cancer_data) #besides ID, diagnosis, and X, every variable is numeric. Change diagnosis to factor (will be our classifier)

#looking at the mean for each variable
lapply(cancer_data[1:33], mean) 
#can see that the means vary goes from 0.104 to 91.97 (can be a problem so we must scale it)


#looking into data a bit more (hypothesized correlations/relationships)
plot(cancer_data$compactness_mean, cancer_data$radius_mean) #see that low compactness has smaller radius 
plot(cancer_data$compactness_mean, cancer_data$smoothness_mean) #low compactness = low smoothness, gradually increases then dissipates
cor(cancer_data$compactness_mean, cancer_data$radius_mean) #0.5 (not strongly correlated)
cor(cancer_data$compactness_mean, cancer_data$smoothness_mean) #0.65 somewhat correlated but not too strong

#pre-processing ####
sum(is.na(cancer_data)) #sum of total missing values, we can see that there are 569 it could be from the X column with 569 missing values

#removing id and X variables 
cancer_data = subset(cancer_data, select = -id)
cancer_data = subset(cancer_data, select = -X)
str(cancer_data) #check to see if they were removed 

#changing variable diagnosis to factor
cancer_data$diagnosis = as.factor(cancer_data$diagnosis) #change diagnosis to factor variable 
str(cancer_data) #check to see if diagnosis is changed to factor 

sum(is.na(cancer_data)) #checking to see how many NA's remain have been removed after removing X variable 


#diagnosis is categorical so we need to remove it for now to do clustering 
cancer_data_no_diagnosis = subset(cancer_data, select = -diagnosis)
str(cancer_data_no_diagnosis) #now that we have all numerical values, scale the data

#noticed in investigating our data that the means range differently 
#need to scale/normalize the data; normalizing the data allows us to make better measurements 
d.scale = data.frame(scale(cancer_data_no_diagnosis)) 

#get the distance matrix between data (getting the distances between the variables)
d.dist = factoextra::get_dist(d.scale, method = 'pearson')
d.dist

#tells us if there are certain parts of the data that naturally cluster together 
factoextra::fviz_dist(d.dist) #there's a lot of data, so we need to look more further into this with k-means clustering


#k-means clustering ####

#1. identify or find optimal k clusters 
#optimal k is where the elbow point does not significantly reduce wss ; so 4-5-6 clusters may be best 
factoextra::fviz_nbclust(d.scale, kmeans, method = 'wss')

#2. identify the 5-7 clusters using k-means

#identifying 5 centers 
k5 = kmeans(d.scale, centers = 5) #d.scale is cancer_data_no_diagnosis that is scaled
k5$size #gives us size of the 4 clusters 
k5$tot.withinss #total wss ; 8709.396
k5$cluster

fviz_cluster(k5, data = d.scale) #visualize the k5 cluster with d.scale data
#we can see there are some distinct clusters but also some overlap between the clusters
cluster::clusplot(d.scale, k5$cluster)

c1 = cancer_data[k5$cluster==1,] #slice/subset data set to see where the cluster value is 1 
c2 = cancer_data[k5$cluster==2,]
c3 = cancer_data[k5$cluster==3,]
c4 = cancer_data[k5$cluster==4,]
c5 = cancer_data[k5$cluster==5,]

#now that we have our clusters, we can perform classification to find any trends

#Classification ####

#before creating training and test set we must create the class attribute
table(c1$diagnosis)
table(c2$diagnosis)
table(c3$diagnosis)
table(c4$diagnosis)
table(c5$diagnosis)

class.attribute = ifelse(k5$cluster == 1, 'high_B',
                         ifelse(k5$cluster == 5, 'high_B',
                                ifelse(k5$cluster == 2, 'high_M',
                                       ifelse(k5$cluster == 3, 'medium_M',
                                              ifelse(k5$cluster == 4, 'medium_M', 'all_the_others')))))



#check data type for class.attribute
class(class.attribute) #need to convert to factor 

cancer_data = data.frame(cancer_data,class.attribute) #adding "class.attribute" into data set
str(cancer_data)


#need to change or remove diagnosis since it is similar to class.attribute 
d1 = subset(cancer_data, select = -diagnosis) # removes diagnosis from the data frame
str(d1)

#convert class attribute to factor
d1$class.attribute = as.factor(d1$class.attribute)
str(d1)

#Creating train/test sets and test attribute ####

#training set
train.index = sample(1:nrow(d1), .8*nrow(d1)) #use 80% of data for training
train.index
train.set = d1[train.index,] #create train set variable 
train.set #view train set 

#test set
test.set = d1[-train.index,] #excludes training set 
test.set 

#creating our test attribute (class attribute)
cancer.test = d1$class.attribute[-train.index]
cancer.test


#train model ####
dtree = tree(class.attribute~., train.set) #decision tree with train set
dtree
plot(dtree) #plot tree
text(dtree) #tree diagram with context (variables visible)
summary(dtree) #get info about dtree 

#testing the trained model ####
dtree.test = predict(dtree, test.set, type = 'class') #would like to return as class lable
table(dtree.test,cancer.test) #confusion matrix 

#Cross-validation
cv.tree(dtree, FUN = prune.misclass) #find what is the most optimal node for our tree


#pruning the trained model 
dtree.pruned = prune.misclass(dtree, best = 6) #best tree of size 6
plot(dtree.pruned)
text(dtree.pruned)
dtree.pruned
summary(dtree.pruned)

#test the pruned tree 
dtree.pruned.test = predict(dtree.pruned, test.set, type='class')
table(cancer.test, dtree.pruned.test)


#Random Forest ####

#training set with Random Forest
dtree.rf = randomForest(class.attribute~., train.set) #train model with random forest
dtree.rf

#testing the trained model with Random Forest 
dtree.rf.test = predict(dtree.rf, test.set, type='class')
table(cancer.test, dtree.rf.test) #confusion matrix
