import numpy as np
from utils.transforms import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


TRANSFORMATIONS = ['Bri', 'Col', 'Con', 'Sha', 'Rot', 'She', 'Tra', 'Aut', 'Equ', 'Inv', 'Pos', 'Sol', 'flip']

def intersection_over_union(boxA, boxB):
    # Extract the coordinates of the two boxes
    topA, leftA, bottomA, rightA = boxA
    topB, leftB, bottomB, rightB = boxB
    
    # Determine the coordinates of the intersection rectangle
    interTop = max(topA, topB)
    interLeft = max(leftA, leftB)
    interBottom = min(bottomA, bottomB)
    interRight = min(rightA, rightB)
    
    # Compute the area of the intersection rectangle
    interWidth = max(0, interRight - interLeft)
    interHeight = max(0, interBottom - interTop)
    interArea = interWidth * interHeight
    
    # Compute the area of both bounding boxes
    boxAArea = (rightA - leftA) * (bottomA - topA)
    boxBArea = (rightB - leftB) * (bottomB - topB)
    
    # Compute the Intersection over Union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou


def lvl2strength(trans, lvl):
    if trans in ['Bri', 'Col', 'Con', 'Sha', 'Rot', 'She', 'Tra']:
        val = 2*lvl - 1
    elif trans in ['Aut', 'Equ', 'Inv']:
        val = 1
    elif trans == 'Pos':
        val = 0.8*lvl + 0.2
    elif trans == 'Sol':
        val = 1 - lvl/256
    else:
        return trans, lvl
    return trans, val


def parts2vector(x1_x2):
    x1, x2 = x1_x2
    
    vec1, vec2 = np.zeros((len(TRANSFORMATIONS),)), np.zeros((len(TRANSFORMATIONS),))
    
    x1 = list(  map( lambda e: lvl2strength(e[0], e[1]) , x1 )  )
    x2 = list(  map( lambda e: lvl2strength(e[0], e[1]) , x2 )  )
    
    for trans, val in x1:
        if trans in TRANSFORMATIONS:
            vec1[TRANSFORMATIONS.index(trans)] = val
    
    for trans, val in x2:
        if trans in TRANSFORMATIONS:
            vec2[TRANSFORMATIONS.index(trans)] = val
        
    # print(vec1, vec2)
    # return  sum(abs(vec1)) + sum(abs(vec2))
    
    iou = intersection_over_union(x1[-2][1], x2[-2][1])

    return  np.concatenate((
        abs(vec1 - vec2),
        # vec1, vec2,  
        [1-iou]
    ))


def get_hardness_estimator(vectors, sims, fraction=0.9, logs=True, plot=True):
    
    s = int(len(vectors) * fraction)    
    X_train_i, y_train_i, X_test_i, y_test_i = vectors[:s], sims[:s], vectors[s:], sims[s:]

    model = LinearRegression()
    # model = MLPRegressor(hidden_layer_sizes=(512,512), max_iter=100, random_state=42, activation='relu')
    # model = MLPRegressor(
    #     hidden_layer_sizes=(512, 512),
    #     max_iter=1000,
    #     random_state=42,
    #     activation='relu',
    #     early_stopping=True,
    #     validation_fraction=0.2,
    #     n_iter_no_change=10,
    #     verbose=True)
    model.fit(X_train_i, y_train_i)


    pred_train_i = model.predict(X_train_i)
    pred_test_i  = model.predict(X_test_i)

    
    if logs:
        print('train loss:', abs(y_train_i - pred_train_i).mean())
        print('test loss:', abs(y_test_i - pred_test_i).mean())
        print("Correlation train:", np.corrcoef(pred_train_i, y_train_i)[0, 1])
        print("Correlation test:", np.corrcoef(pred_test_i, y_test_i)[0, 1])

    if plot:
        plt.scatter(pred_train_i, y_train_i, marker='.', c='blue', alpha=1)
        plt.scatter(pred_test_i, y_test_i, marker='.', c='green', alpha=1)
        plt.xlabel('Estimation')
        plt.ylabel('Sims')
        plt.title('Scatter Plot with Best-Fit Line')
        plt.show()
    
    return model