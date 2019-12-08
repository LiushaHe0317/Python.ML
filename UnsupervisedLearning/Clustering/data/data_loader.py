import numpy
from scipy.sparse import csr_matrix


class DataLoader:
    """
    This class loads data file.
    """
    def load_npz_file(self, filename):
        """
        This method loads npz file.

        :param filename: A sting of path to data file.
        :return: A ``scipy.csr_matrix`` object.
        """
        loader = numpy.load(filename)

        data = loader['data']
        indices = loader['indices']
        indptr = loader['indptr']
        shape = loader['shape']

        return csr_matrix((data, indices, indptr), shape)