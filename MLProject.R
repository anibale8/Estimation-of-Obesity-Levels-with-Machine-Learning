#MACHINE LEARNING PROJECT - ANIBAL HERNANDO NOVO 907977

# Clean the environment
rm(list=ls()); graphics.off(); cat("\014")

# Load libraries 
library(dplyr)
library(tidyr)
library(readr)
library(e1071)
library(caret)
library(randomForest)
library(keras)
library(tensorflow)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(RColorBrewer)
library(clValid)
library(EMCluster)
library(mclust)
library(dbscan)
library(cluster)
library(factoextra)
library(rpart.plot)

obesity <- read_csv("C:/ANIBALE/UNIVERSIDAD/CURSO 4º BICOCCA/2º Cuatri/MACHINE LEARNING M/Project ML/estimation+obesity/ObesityData.csv")

head(obesity)

obesity <- obesity[,-c(3,4)]

#############PREPROCESSING##############

# Variable 'label' balanced
freq <- table(obesity$NObeyesdad) 

percent <- prop.table(freq)

freq
percent

# Classes are balanced

# Convert binary variables to numeric factors
obesity <- obesity %>%
  mutate(
    Gender = ifelse(Gender == "Male", 1, 0),
    family_history_with_overweight = ifelse(family_history_with_overweight == "yes", 1, 0),
    FAVC = ifelse(FAVC == "yes", 1, 0),
    SMOKE = ifelse(SMOKE == "yes", 1, 0),
    SCC = ifelse(SCC == "yes", 1, 0)
  )

# Mapping for non-binary categorical variables
caec_mapping <- c("no" = 0, "Sometimes" = 1, "Frequently" = 2, "Always" = 3)
calc_mapping <- c("no" = 0, "Sometimes" = 1, "Frequently" = 2, "Always" = 3)
mtrans_mapping <- c("Walking" = 0, "Bike" = 1, "Motorbike" = 2, "Public_Transportation" = 3, "Automobile" = 4)
nobeyesdad_mapping <- c(
   "Insufficient_Weight" = 0,
   "Normal_Weight" = 1,
   "Overweight_Level_I" = 2,
   "Overweight_Level_II" = 3,
   "Obesity_Type_I" = 4,
   "Obesity_Type_II" = 5,
   "Obesity_Type_III" = 6
 )

# Apply mappings to corresponding columns
obesity$CAEC <- caec_mapping[obesity$CAEC]
obesity$CALC <- calc_mapping[obesity$CALC]
obesity$MTRANS <- mtrans_mapping[obesity$MTRANS]
obesity$NObeyesdad <- nobeyesdad_mapping[obesity$NObeyesdad]

# Change the type of numeric variables to integer, because
# if they are kept as double, the values cause problems

obesity$Age <- as.integer(obesity$Age)
obesity$Weight <- as.integer(obesity$Weight)
obesity$FCVC <- as.integer(obesity$FCVC)
obesity$NCP <- as.integer(obesity$NCP)
obesity$CH2O <- as.integer(obesity$CH2O)
obesity$FAF <- as.integer(obesity$FAF)
obesity$TUE <- as.integer(obesity$TUE)

names(obesity)[names(obesity) == 'family_history_with_overweight'] <- 'FHWO'

# View the first rows of the transformed dataset
head(obesity)



# Normalize the variables (except the column with obesity level)
normalized_data <- obesity %>% 
  mutate_all(~(.-min(.))/(max(.)-min(.)))

# Convert data to long format for ggplot2
data_long <- normalized_data %>%
  gather(key = "variable", value = "value", -NObeyesdad)


#Boxplot Normalization
ggplot(data_long, aes(x = variable, y = value,color=variable)) +
  geom_violin()+
  theme_minimal() +
  labs(title = "Distribution Normalization",
       x = "Variable",
       y = "Normalizate Value") +
  theme(axis.text.x = element_text(size= 8, angle = 45, hjust = 1))

# Convert the target variable to factor: ensures that classification models treat
# the classes correctly and that predictions and evaluations are accurate.

obesity$NObeyesdad <- as.factor(obesity$NObeyesdad)

# Variable importance with RF

# Fit a random forest model
set.seed(123) 
modelo_rf <- randomForest(NObeyesdad ~ ., data = obesity, importance = TRUE)

# Show the importance of variables
importancia_variables <- importance(modelo_rf, type = 2) # type = 2 uses the Gini index
print(importancia_variables)

# Visualize the importance of variables
varImpPlot(modelo_rf, type = 2, main = "Variable Importance")



#############CLUSTERING#################

# Remove the label variable: Nobeyesdad
obesClust <- obesity[,-15]

# Standardize the variables
obesClust <- as.data.frame(scale(obesClust))



###Probabilistic model-based Clustering - EM Cluster###

set.seed(127)
em_result <- Mclust(obesClust,assign.class=T)

# View model summary
summary(em_result)

# View assigned clusters, assuming 'em_result' can be plotted directly
plot(em_result, "BIC", xlab = "Número de Clusters", ylab = "Criterio BIC",ylim = c(-100000,-40000))

# We can see that the highest BIC model is VEV with 7 clusters.
# Let's use different methods to see the number of clusters to use.


#1.SSQ- Elbow graph to know the number of clusters (k) to use

set.seed(123)
SSQs_list <- list()
kappas <- 2:10

# Repeat 50 times
for (i in 1:50) {
  SSQs <- numeric()
  for (k in kappas) {
    km.res <- kmeans(obesClust, centers = k)
    SSQs <- c(SSQs, km.res$tot.withinss)
  }
  SSQs_list[[i]] <- SSQs
}

# Calculate the mean of the total sum of squares within clusters for each k
SSQs_mean <- sapply(1:length(kappas), function(i) mean(sapply(SSQs_list, `[`, i)))

# Plot the results
plot(kappas, SSQs_mean, type = "b", xlab = "Number of Clusters (k)", ylab = "SSQ Mean within Clusters", main = "SSQ Plot",ylim = c(18000,32000))

# We can see that the elbow of the plot is not clearly visible, but it can be between 6 and 8.


#2.clVal package - Silhoutte with k-means

set.seed(123)
sils <- numeric()
kappas <- 2:10
for( k in kappas ) {
  res <- kmeans(obesClust, centers=k )
  sil <- silhouette( res$cluster, dist(obesClust) )
  plot(sil,main=paste("Clusters =",k),col = 2:k, border = NA)
  abline(v=mean(sil[,3]),col="red",lty=2,lwd=2)
  sils <- c(sils, (summary(sil))$avg.width )
}

plot( kappas, sils, type="o", lwd=3, col="red" )

# We can see that in all clusters there is some point with
# negative silhouette, so we cannot accept this method.


###K-Means###

set.seed(123)  
kmeans_result <- kmeans(obesClust, centers = 7)

# Define custom names for each cluster
cluster_names <- c("Insufficient Weight", "Normal Weight", "Overweight Level I",
                   "Overweight Level II", "Obesity Type I", "Obesity Type II",
                   "Obesity Type III")

# Sub dataset with height y weight
k_data<-cbind(obesClust$Weight,obesClust$Height)
colnames(k_data) <- c("Weight","Height")
plot(k_data)
clrs <- rainbow(7)
for( i in 1:7 ) {
  points(k_data[which(kmeans_result$cluster==i),,drop=F],col=clrs[i],pch=19)
}
legend("bottomright", legend = 1:7, col = clrs, pch = 19, cex=0.9, title = "Cluster")

# Plot centroids
centers <- cbind(kmeans_result$centers[,4],kmeans_result$centers[,3])
points(centers, col="black", pch=22, cex=2, bg=clrs)

# Confusion matrix
tabla_contingencia <- table(kmeans_result$cluster, obesity$NObeyesdad)

# Calculate percentages instead of absolute frequencies
df_tabla <- as.data.frame(tabla_contingencia)
df_tabla$Percentage <- df_tabla$Freq / rowSums(tabla_contingencia) * 100

# Create the bar plot with percentages instead of frequencies
ggplot(df_tabla, aes(Var1, Percentage, fill=Var2)) +
  geom_bar(stat="identity", position="dodge") +
  labs(x="Cluster", y="Percentage", fill="Obesity Level") +  
  scale_fill_manual(values = brewer.pal(7, "Set3"),  
                    labels = cluster_names) +
  labs(title = "K-Means Clustering") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


###Hierarchical Clustering###

# Distance matrix
set.seed(124)
d <- dist(obesity)

# Perform hierarchical clustering
hc_result <- hclust(d, method = "complete")

# Drow the dendrograma
plot(hc_result, labels = FALSE, hang = -1)

# Drow the clusters
rect.hclust(hc_result, k = 3, border = 2:4)

k <- 3
clusters <- cutree(hc_result, k)

hc_clustering <- cbind(predicted = as.factor(clusters),levels = obesity$NObeyesdad)

# hc_clustering to dataframe
hc_clustering <- as.data.frame(hc_clustering)
colnames(hc_clustering) <- c("predicted", "levels")

# Convert to factor
hc_clustering$predicted <- as.factor(hc_clustering$predicted)
hc_clustering$levels <- as.factor(hc_clustering$levels)

# Amount in each level
obesity_distribution <- hc_clustering %>%
  group_by(predicted, levels) %>%
  summarise(Count = n()) %>%
  mutate(Percentage = Count / sum(Count) * 100) %>%
  ungroup()

# Barplot of obesity distribution
ggplot(obesity_distribution, aes(x = predicted, y = Percentage, fill = levels)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  scale_fill_manual(values = brewer.pal(7, "Set3"),
                    labels = cluster_names) +
  labs(title = "Hierarchical Clustering",
       x = "Predicted Cluster",
       y = "Percentage",
       fill = "Obesity Level")



###DBSCAN###

# Algorithm to check the best values of DBSCAN using silhouette
set.seed(126)
sils <- numeric()
eps_values <- seq(0.3, 0.9, by = 0.01)
minPts_values <- c(5:30)

# Vectors to store the best values of eps and minPts
best_eps <- NA
best_minPts <- NA
best_silhouette <- -Inf

# Iterate over different values of eps and minPts
for (eps in eps_values) {
  for (minPts in minPts_values) {
    # Apply DBSCAN to scaled data
    dbscan_result <- dbscan(obesClust, eps = eps, minPts = minPts)
    
    # Calculate the silhouette coefficient
    sil <- silhouette(dbscan_result$cluster, dist(obesClust))
    
    # Calculate the average silhouette width
    avg_silhouette <- mean(sil[, 3])
    
    # Store the values of eps and minPts if the average silhouette is greater than the best found so far
    if (avg_silhouette > best_silhouette) {
      best_eps <- eps
      best_minPts <- minPts
      best_silhouette <- avg_silhouette
    }
    
    # Store the average silhouette width
    sils <- c(sils, avg_silhouette)
  }
}

# Plot the results
plot(sils, type = "o", lwd = 3, col = "red", xlab = "Index", ylab = "Silhouette Width", main = "Silhouette Width vs. Index")

# Find the index of the maximum silhouette value
indice_max_sil <- which.max(sils)

# Find the corresponding values of eps and minPts
mejor_eps <- eps_values[((indice_max_sil - 1) %% length(eps_values)) + 1]
mejor_minPts <- minPts_values[((indice_max_sil - 1) %/% length(eps_values)) + 1]

# Show values
cat("Mejor valor de eps:", mejor_eps, "\n")
cat("Mejor valor de minPts:", mejor_minPts, "\n")
cat("Mejor valor de silhouette:", best_silhouette, "\n")

dbscan_best_result<- dbscan(obesClust, eps = mejor_eps, minPts = mejor_minPts)
dbscan_best_result

# Get clusters and assignments
clusters <- dbscan_best_result$cluster
clusters <- factor(clusters)


# Contingency table
tabla_contingencia <- table(clusters, obesity$NObeyesdad)

# Convert table to data frame and calculate percentages
df_tabla <- as.data.frame(tabla_contingencia)
df_tabla$Percentage <- df_tabla$Freq / rowSums(tabla_contingencia) * 100

# Create bar plot with percentages
ggplot(df_tabla, aes(x = clusters, y = Percentage, fill = Var2)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Cluster", y = "Percentage", fill = "Obesity Level") +
  scale_fill_manual(values = brewer.pal(7, "Set3"),  
                    labels = cluster_names) +  
  labs(title = "DBSCAN Clustering") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))



#############CLASIFICATION##############

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(obesity$NObeyesdad, p = .8, 
                                  list = FALSE, 
                                  times = 1)
obesityTrain <- obesity[ trainIndex,]
obesityTest  <- obesity[-trainIndex,]


# Method 1: K-Nearest Neighbors (KNN)
# Define the training control with cross-validation
ctrl <- trainControl(method = "cv", number = 5)      

# Train the KNN model with cross-validation
knnModel <- train(NObeyesdad ~ ., data = obesityTrain, method = "knn", trControl = ctrl)
knnPred <- predict(knnModel, newdata = obesityTest)
knnConfMatrix <- confusionMatrix(knnPred, obesityTest$NObeyesdad)

# Evaluate the model on the test set to calculate the generalization error
knn_generalization_error <- mean(knnPred != obesityTest$NObeyesdad)

# Calculate the average empirical error during cross-validation
knn_empirical_error <- 1 - mean(knnModel$resample$Accuracy)

# Show results
print(paste("Generalization Error:", knn_generalization_error))
print(paste("Empirical Error:", knn_empirical_error))


# Method 2: Support Vector Machine (SVM)
set.seed(123)
svmModel <- train(NObeyesdad ~ ., data = obesityTrain, method = "svmRadial",trControl = ctrl)
svmPred <- predict(svmModel, newdata = obesityTest)
svmConfMatrix <- confusionMatrix(svmPred, obesityTest$NObeyesdad)

# Generalization error
svm_generalization_error <- mean(svmPred != obesityTest$NObeyesdad)

# Calculate the average empirical error during cross-validation
svm_empirical_error <- 1 - mean(svmModel$resample$Accuracy)

# Show results
print(paste("Generalization Error:", svm_generalization_error))
print(paste("Empirical Error:", svm_empirical_error))


# Method 3: Decision Tree
set.seed(123)
treeModel <- train(NObeyesdad ~ ., data = obesityTrain, method = "rpart",trControl = ctrl)
treePred <- predict(treeModel, newdata = obesityTest)
treeConfMatrix <- confusionMatrix(treePred, obesityTest$NObeyesdad)

# Generalized Error
tree_generalization_error <- mean(treePred != obesityTest$NObeyesdad)

# Calculate the average empirical error during cross-validation
tree_empirical_error <- 1 - mean(treeModel$resample$Accuracy)

# Show results
print(paste("Generalization Error:", tree_generalization_error))
print(paste("Empirical Error:", tree_empirical_error))

# Plot the decision tree
rpart.plot(treeModel$finalModel)


# Method 4: Random Forest
set.seed(123)
rfModel <- train(NObeyesdad ~ ., data = obesityTrain, method = "rf",trControl = ctrl)
rfPred <- predict(rfModel, newdata = obesityTest)
rfConfMatrix <- confusionMatrix(rfPred, obesityTest$NObeyesdad)

# Generalization error
rf_generalization_error <- mean(rfPred != obesityTest$NObeyesdad)

# Calculate the average empirical error during cross-validation
rf_empirical_error <- 1 - mean(rfModel$resample$Accuracy)

# Show results
print(paste("Generalization Error:", rf_generalization_error))
print(paste("Empirical Error:", rf_empirical_error))

# View the number of trees in the fitted model
num_trees <- rfModel$finalModel$ntree
print(paste("Número de árboles en el modelo es", num_trees))

# Imprimir las matrices de confusión
print(knnConfMatrix)
print(svmConfMatrix)
print(treeConfMatrix)
print(rfConfMatrix)







