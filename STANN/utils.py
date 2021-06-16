"""
   ___                  _                
  / _/______ ____  ____(_)__ _______     
 / _/ __/ _ `/ _ \/ __/ (_-</ __/ _ \    
/_//_/  \_,_/_//_/\__/_/___/\__/\___/    
  ___ _____(_)__ ___ ____  / /_(_)       
 / _ `/ __/ (_-</ _ `/ _ \/ __/ /        
 \_, /_/ /_/___/\_,_/_//_/\__/_/         
/___/

#chequeando que funciona!

Samee Lab @ Baylor College Of Medicine
francisco.grisanticanozo@bcm.edu
Date: 12/2019

"""
import scanpy as sc
import pandas as pd
import sklearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def print_model(choice):
    """
    prints model summary or plot
    
    """
    if choice == 'summary':    
        model.summary()
    elif choice == 'plot':
        keras.utils.plot_model(model,to_file='model_plot.png',show_shapes=True, show_layer_names=True)
    else:
        print("choose 'summary' or 'plot' output")



def organize_data(adata_train=None,adata_predict=None):
    
    Y_train=None
    
    try:
        
        
        X_train = adata_train.to_df()
        X_predict = adata_predict.to_df()
        Y_train = pd.DataFrame(adata_train.obs.celltype.copy()).values
        
        
        if (list(X_train.columns) != list(X_predict.columns)):
            gene_list = list(X_train.columns)
            indices_sc = list(adata_train.to_df().index)
            Y_train = pd.DataFrame(adata_train.obs.celltype.copy()).values
            X_train = adata_train.to_df()
            X_train = X_train[gene_list]
            X_predict = adata_predict.to_df()
            indices_predict = list(adata_predict.to_df().index)
            X_predict = X_predict[gene_list]
            
            print(f'[INFO] Equal columns = {list(X_train.columns) == list(X_predict.columns)}')
            print("[INFO] Data organized")
            
    except:
        print("Failed -- Data must be same length")
    
    return X_train, Y_train, X_predict

def min_max(X=None):
    X = X.values.astype(float)
    scaler = sklearn.preprocessing.MinMaxScaler()
    X_tranformed = scaler.fit_transform(X)
    return X_tranformed,scaler

def label_encoder(Y_train=None):
    
    encoder = sklearn.preprocessing.LabelEncoder()
    encoder.fit(Y_train)
    
    Y_train_dummy = encoder.transform(Y_train)
    Y_train_ohe = tf.keras.utils.to_categorical(Y_train_dummy)
    
    return Y_train_dummy,Y_train_ohe,encoder


def get_train_test_split(X_sc=None, 
                     encoded_y_sc=None, 
                     test_size=0.10, 
                     random_state=40):
    
    #Generating train-test split
    X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(X_sc, 
                                                                                        encoded_y_sc, 
                                                                                        test_size=0.10, 
                                                                                        random_state=40)

    print(f'[INFO] X_train shape={X_train_sc.shape}')
    print(f'[INFO] y_train shape={y_train_sc.shape}')
    print(f'')
    print(f'[INFO] X_test shape={X_test_sc.shape}')
    print(f'[INFO] y_test shape={y_test_sc.shape}')
    
    return X_train_sc, X_test_sc, y_train_sc, y_test_sc

def get_class_weights(Y_train_ohe=None):
    class_weights = class_weight.compute_class_weight('balanced' ,np.unique(np.argmax(Y_train_ohe, axis=1)) ,np.argmax(Y_train_ohe, axis=1))
    return class_weights

def print_metrics(model=None,
                  x_train=None,
                  y_train=None,
                  x_test=None,
                  y_test=None):
    
    # evaluate the model accuracy
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('[INFO] Accuracy -- Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    # evaluate the model roc
    train_roc = sklearn.metrics.roc_auc_score(y_train, model.predict(x_train))
    test_roc = sklearn.metrics.roc_auc_score(y_test, model.predict(x_test))

    print('[INFO] ROC -- Train: %.3f, Test: %.3f' % (train_roc , test_roc))

def get_metrics(model=None,
                  x_train=None,
                  y_train=None,
                  x_test=None,
                  y_test=None):
    """
    Returns

    train_acc, test_acc, train_roc, test_roc

    """
    # evaluate the model accuracy
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # evaluate the model roc
    train_roc = sklearn.metrics.roc_auc_score(y_train, model.predict(x_train))
    test_roc = sklearn.metrics.roc_auc_score(y_test, model.predict(x_test))
    
    return train_acc, test_acc, train_roc, test_roc

def make_predictions(model=None,
                     X_predict=None,
                     encoder=None,
                     adata_predict=None,
                     probabilities=False,
                     save=True
                    ):
    
    y_pred = model.predict_classes(X_predict)
    y_pred = encoder.inverse_transform(y_pred)
    
    predictions = adata_predict.obs.copy()
    predictions['STANN_predictions'] = y_pred
    
    if not os.path.exists('../outputs/'):
        os.makedirs('../outputs')
    
    if save == True:
        predictions.to_csv('../outputs/predictions.csv')
        
    return predictions

def cross_validate(n_folds=10,X=None,y=None,model=None):
    """
    returns and print cross valditation scores
    
    parameters-> 
    
    n_folds = 10 (predetermined)
    X= X and y=y (data)
    model = keras model
    
    """
    
    cv_scores, model_history = list(), list()
    
    report = {}
    
    for fold in range(n_folds):
        
        print(f'For fold = {fold}')
              
        
        tuned_hp = {'first_dense': 160,
                     'dense_activation': 'tanh',
                    'second_dense': 145.0,
                     'learning_rate': 0.01}
        
        # split data
        
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size=0.10, random_state = np.random.randint(1,1000, 1)[0])
        
        #creates model
        model = None # Clearing the NN.
        model = create_model(act_fun=tuned_hp['dense_activation'],
                     first_dense=tuned_hp['first_dense'],
                     second_dense=tuned_hp['second_dense'],
                     learning_rate=tuned_hp['learning_rate'],
                     input_dim=X_train.shape[1],
                     output_dim=y_train.shape[1])
        
        #simple early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', verbose=1,patience=30)
        #compute class weights
        class_weights = sklearn.utils.class_weight.compute_class_weight('balanced' ,np.unique(np.argmax(y_train, axis=1)) ,np.argmax(y_train, axis=1))
        
        #fit the keras model on the dataset
    
        history = model.fit(X_train, 
                    y_train, 
                    validation_split=0.10,
                    epochs=30,
                    class_weight=class_weight,
                    callbacks=[es])
        
        #get scoring metrics
        _, train_acc = model.evaluate(X_train, y_train, verbose = 0)
        _, test_acc = model.evaluate(X_val, y_val, verbose=0)
        
        # evaluate model
        #model, test_acc = evaluate_model(X_train, X_val, y_train, y_val)
        

        print(f'>train_acc={train_acc} & test_acc={test_acc}')
        print('')
        
        cv_scores.append([train_acc, test_acc])
        model_history.append(history)
        
        report_train = sklearn.metrics.classification_report(y_train_sc_decoded,y_train_pred_sc,output_dict=True)
        report_train = pd.DataFrame(report_train)
        report_train['test'] = test
        report_train_all = report_train_all.append(report_train)
    
    return cv_scores, model_history

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          size=(20,20)):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


    