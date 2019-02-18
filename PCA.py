import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import  nn_arch,nn_reg
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

"""
Thanks to jtay for the code - https://github.com/JonathanTay/CS-7641-assignment-3

Dimensionality Reduction using Principal Component Analysis

"""

def main():

    out = './BASE/'
    cmap = cm.get_cmap('Spectral')

    np.random.seed(0)
    letter = pd.read_hdf('./BASE/datasets.hdf', 'letter')
    letterX = letter.drop('Class',1).copy().values
    letterY = letter['Class'].copy().values

    madelon = pd.read_hdf('./BASE/datasets.hdf','madelon')
    madelonX = madelon.drop('Class',1).copy().values
    madelonY = madelon['Class'].copy().values


    madelonX = StandardScaler().fit_transform(madelonX)
    letterX= StandardScaler().fit_transform(letterX)

    clusters =  [2,5,10,15,20,25,30,35,40]
    dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]

    dims2 = [2,4,6,8,10,12,14,16]
    #5, 10, raise
    #%% data for 1

    pca = PCA(random_state=5)
    pca.fit(madelonX)
    tmp = pd.Series(data = pca.explained_variance_,index = range(1,501))
    tmp.to_csv(out+'madelonPCA.csv')


    pca = PCA(random_state=5)
    pca.fit(letterX)
    tmp = pd.Series(data = pca.explained_variance_,index = range(1,17))
    tmp.to_csv(out+'letterPCA.csv')


    #%% Data for 2

    grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    pca = PCA(random_state=5)
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('pca',pca),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(madelonX,madelonY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'Madelon dim red.csv')


    grid ={'pca__n_components':dims2,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    pca = PCA(random_state=5)
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('pca',pca),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(letterX,letterY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'letter dim red.csv')
    # raise
    #%% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
    dim = 5
    pca = PCA(n_components=dim,random_state=10)

    madelonX2 = pca.fit_transform(madelonX)
    madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(madelonY).T)))
    cols = list(range(madelon2.shape[1]))
    cols[-1] = 'Class'
    madelon2.columns = cols
    madelon2.to_hdf(out+'datasets.hdf','madelon',complib='blosc',complevel=9)

    dim = 14
    pca = PCA(n_components=dim,random_state=10)
    letterX2 = pca.fit_transform(letterX)
    letter2 = pd.DataFrame(np.hstack((letterX2,np.atleast_2d(letterY).T)))
    cols = list(range(letter2.shape[1]))
    cols[-1] = 'Class'
    letter2.columns = cols
    letter2.to_hdf(out+'datasets.hdf','letter',complib='blosc',complevel=9)

if __name__ == '__main__':
    main()