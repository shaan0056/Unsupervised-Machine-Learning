import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM, nn_arch, nn_reg
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import homogeneity_score as hs,completeness_score as cs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
import sys
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
import time as time

"""
Thanks to jtay for the code - https://github.com/JonathanTay/CS-7641-assignment-3

Create K means and EM GMM clusters

"""

def main_logic():

    out = './BASE/'
    base = './PCA/'
    np.random.seed(0)
    letter = pd.read_hdf(base + 'datasets.hdf', 'letter')
    letter_X = letter.drop('Class', 1).copy().values
    letter_Y = letter['Class'].copy().values

    madelon = pd.read_hdf(base + 'datasets.hdf', 'madelon')
    madelon_X = madelon.drop('Class', 1).copy().values
    madelon_Y = madelon['Class'].copy().values

    np.random.seed(0)
    madelon_X = StandardScaler().fit_transform(madelon_X)
    letter_X = StandardScaler().fit_transform(letter_X)
    
    clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    
    # %% Data for 1-3
    SSE = defaultdict(dict)
    ll = defaultdict(dict)
    Silhouette_dict = defaultdict(dict)
    acc = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)
    

    for k in clusters:
        st = clock()
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(madelon_X)
        gmm.fit(madelon_X)

        SSE[k]['Madelon'] = km.score(madelon_X)
        ll[k]['Madelon'] = gmm.score(madelon_X)
        test = km.predict(madelon_X)
        acc[k]['Madelon']['Kmeans'] = cluster_acc(madelon_Y, km.predict(madelon_X),k)
        acc[k]['Madelon']['GMM'] = cluster_acc(madelon_Y, gmm.predict(madelon_X))
        adjMI[k]['Madelon']['Kmeans'] = ami(madelon_Y, km.predict(madelon_X))
        adjMI[k]['Madelon']['GMM'] = ami(madelon_Y, gmm.predict(madelon_X))
        print ("Homogenity Score ,{}, Kmeans,".format(k),hs(madelon_Y,km.labels_))
        print("Completeness Score ,{} ,Kmeans,".format(k), cs(madelon_Y, km.labels_))
        label = km.labels_
        gmmm = gmm.predict_proba(madelon_X)
        sil_coeff = silhouette_score(madelon_X, label, metric='euclidean')
        Silhouette_dict[k]['Madelon'] = sil_coeff
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))

        km.fit(letter_X)
        gmm.fit(letter_X)
        SSE[k]['letter'] = km.score(letter_X)
        ll[k]['letter'] = gmm.score(letter_X)
        best = km.predict(letter_X)
        acc[k]['letter']['Kmeans'] = cluster_acc(letter_Y, km.predict(letter_X),k)
        acc[k]['letter']['GMM'] = cluster_acc(letter_Y, gmm.predict(letter_X))
        adjMI[k]['letter']['Kmeans'] = ami(letter_Y, km.predict(letter_X))
        adjMI[k]['letter']['GMM'] = ami(letter_Y, gmm.predict(letter_X))
        label = km.labels_
        sil_coeff = silhouette_score(letter_X, label, metric='euclidean')
        Silhouette_dict[k]['letter'] = sil_coeff
        print("Homogenity Score ,{}, Kmeans,".format(k), hs(letter_Y, km.labels_))
        print("Completeness Score ,{} ,Kmeans,".format(k), cs(letter_Y, km.labels_))
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))
        print(k, clock() - st)

    Silhouette_dict = pd.DataFrame(Silhouette_dict).to_csv(out + 'Silhouette.csv')
    SSE = (-pd.DataFrame(SSE)).T
    SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
    ll = pd.DataFrame(ll).T
    ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)
    acc = pd.Panel(acc)
    adjMI = pd.Panel(adjMI)

    SSE.to_csv(out + 'SSE.csv')
    ll.to_csv(out + 'logliklihood.csv')
    acc.ix[:, :, 'letter'].to_csv(out + 'letter acc.csv')
    acc.ix[:, :, 'Madelon'].to_csv(out + 'Madelon acc.csv')
    adjMI.ix[:, :, 'letter'].to_csv(out + 'letter adjMI.csv')
    adjMI.ix[:, :, 'Madelon'].to_csv(out + 'Madelon adjMI.csv')

    # %% NN fit data (2,3)
    grid = {'km__n_clusters': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    madelon = pd.read_hdf(base + 'datasets.hdf', 'madelon')
    madelon_X = madelon.drop('Class', 1).copy().values
    madelon_Y = madelon['Class'].copy().values
    X_train, X_test, y_train, y_test = train_test_split(madelon_X, madelon_Y, test_size=0.3, random_state=42)

    np.random.seed(0)

    for k in clusters:
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5,alpha=10 ** -5,hidden_layer_sizes=(62,62),verbose=0)
        km = kmeans(random_state=5,n_clusters=k)
        pipe = Pipeline([('km', km), ('NN', mlp)])
        #gs = GridSearchCV(pipe, grid, verbose=10)
        tick = time.clock()
        pipe.fit(X_train, y_train)
        tock = time.clock() - tick
        print("Traning time , {}, k means dataset".format(k),',', tock)
        tick = time.clock()
        y_pred = pipe.predict(X_test)
        tock = time.clock() - tick
        print("Testing time , {}, k means component".format(k),',', tock)
        print("Accuracy Score ,  {}, kmeans Madelon".format(k),',', accuracy_score(y_test, y_pred))

        grid = {'gmm__n_components': clusters}
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5,verbose=0,alpha=10 ** -5,hidden_layer_sizes=(62,62))
        gmm = myGMM(random_state=43,n_components=k)
        pipe = Pipeline([('gmm', gmm), ('NN', mlp)])
        #gs = GridSearchCV(pipe, grid, verbose=10, cv=5)
        tick = time.clock()
        pipe.fit(X_train, y_train)
        tock = time.clock() - tick
        print("Traning time , {}, gmm dataset".format(k),',', tock)
        tick = time.clock()
        y_pred = pipe.predict(X_test)
        tock = time.clock() - tick
        print("Testing time , {}, gmm means component".format(k),',', tock)
        print("Accuracy Score , {}, gmm means Madelon".format(k),',', accuracy_score(y_test, y_pred))


    
    grid = {'km__n_clusters': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    km = kmeans(random_state=5)
    pipe = Pipeline([('km', km), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10)

    gs.fit(madelon_X, madelon_Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'Madelon cluster Kmeans.csv')

    grid = {'gmm__n_components': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    gmm = myGMM(random_state=5)
    pipe = Pipeline([('gmm', gmm), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(madelon_X, madelon_Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'Madelon cluster GMM.csv')

    grid = {'km__n_clusters': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    km = kmeans(random_state=5)
    pipe = Pipeline([('km', km), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(letter_X, letter_Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'letter cluster Kmeans.csv')

    grid = {'gmm__n_components': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    gmm = myGMM(random_state=5)
    pipe = Pipeline([('gmm', gmm), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(letter_X, letter_Y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'letter cluster GMM.csv')

    # %% For chart 4/5
    madelonX2D = TSNE(verbose=10, random_state=5).fit_transform(madelon_X)
    letter_X2D = TSNE(verbose=10, random_state=5).fit_transform(letter_X)

    madelon2D = pd.DataFrame(np.hstack((madelonX2D, np.atleast_2d(madelon_Y).T)), columns=['x', 'y', 'target'])
    letter2D = pd.DataFrame(np.hstack((letter_X2D, np.atleast_2d(letter_Y).T)), columns=['x', 'y', 'target'])

    madelon2D.to_csv(out + 'madelon2D.csv')
    letter2D.to_csv(out + 'letter2D.csv')



if __name__ == '__main__':


    main_logic()