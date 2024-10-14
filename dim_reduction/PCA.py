from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, FastICA


def do_PCA(x, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(x)


def do_CCA(x, n_components=2):
    cca = CCA(n_components=n_components)
    return cca.fit_transform(x)


def do_ICA(x, n_components=2):
    ica = FastICA(n_components=n_components)
    S = ica.fit_transform(x)
    A = ica.mixing_
    return S, A
