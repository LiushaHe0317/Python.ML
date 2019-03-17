
import numpy as np
from sklearn.model_selection import train_test_split

x, y = np.arange(20).reshape((10,2)), range(10)

x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.3)
