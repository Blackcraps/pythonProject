# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

#------------K-means------------------
#Nous avons fait un K-means et affiché les clsuter pour avoir une bonne représentation du problème pour commencer.
#Nous avons aussi calculé des score comme le silhouette pour mieux comprendre le problème.
del df_base['id_client']
del df_base['type_de_paiement']
del df_base['total_factures']

data = df_base.loc[:, df_base.columns != 'sortie_client']
labels = df_base["sortie_client"]
#en fonction de si c'est un tableau Numpy ou pas il faut prendre la transposée attention
#labels = labels.T
labelsArray = labels.to_numpy()


label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(labelsArray)
print(true_labels)

n_clusters = len(label_encoder.classes_)
print(n_clusters)

preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)

clusterer = Pipeline(
   [
       (
           "kmeans",
           KMeans(
               n_clusters= n_clusters,
               init="k-means++",
               n_init=1,
               max_iter=1000,
               random_state=42,
           ),
       ),
   ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)

pipe.fit(data)

#calcul du score silhouette et de l'ARI
preprocessed_data = pipe["preprocessor"].transform(data)
predicted_labels = pipe["clusterer"]["kmeans"].labels_
silhouette = silhouette_score(preprocessed_data, predicted_labels)
print("silhouette score : ", silhouette)

ari = adjusted_rand_score(labelsArray, predicted_labels)
print("adjusted rand score : ", ari)

#affichage cluster

pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["component_1", "component_2"],
)

pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
pcadf["true_label"] = label_encoder.inverse_transform(labelsArray)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(10, 10))

scat = sns.scatterplot(
    "component_1",
    "component_2",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    style="true_label",
    palette="Set2",
)

scat.set_title(
    "Clustering results"
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()
