import gc
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def reduce_mem_usage(df):
    """ 
    	Função que itera sobre as colunas de um DataFrame e modifica 
    	os tipos das variáveis para economizar memório
    	
    	Inputs: - df (DataFrame a ser modificado):  pandas.DataFrame
    	-------        
    	
    	Outputs: - df (O DataFrame modificado)   :  pandas.DataFrame
    	-------
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in tqdm([x for x in df.columns]):
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            if col in ['id', 'sign','sex','ok_since']:
            	df[col] = df[col].astype("string[pyarrow]")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    
 
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
    
    
## ---


def display_feat_import(X_values, y_values, ax):
    
    """
    Exibe uma grade contendo 4 gráficos 2D referentes as importâncias das variáveis de um problema 
    de classificação dicotômica. Esta função usa três algoritmos diferentes: Logistic Regression, 
    Random Fores Classifier e PCA.
    
    Além disso, ele exibe a variação cumulativa explicatória dada pelo algoritmo PCA,
    e fornece os coeficientes de combinação linear dos componentes principais construídos. 
        
        inputs: - X_values (a 2D array containing features):        pandas.DataFrame
        ------- - y_values (a 1D array containing target classes):  pandas.Series
                - ax (a matplotlib axis grid to plot figures on):   matplotlib.axes._subplots
                
        outputs: - loadings (coefficients of the linear combination
        --------            of the original variables from which the 
                            principal components are constructed):   pandas.DataFrame
    """
        
    scaler = StandardScaler()
    models = [LogisticRegression(),
              RandomForestClassifier(n_estimators=200, max_depth=7),
              PCA()]
    
    for axis, model in zip(ax.reshape(-1)[:3], models):
        
            clf = make_pipeline(StandardScaler(), model)
            clf.fit(X_values, y_values)
            
            if clf.steps[1][0]=='pca':
                loadings = pd.DataFrame(
                    data=clf[1].components_.T * np.sqrt(clf[1].explained_variance_), 
                    columns=[f'PC{i}' for i in range(1, len(X_values.columns) + 1)],
                    index=X_values.columns
                )
                
                pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
                pc1_loadings = pc1_loadings.reset_index()
                pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

                axis.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
                axis.set_title('PCA loading scores (primeira componente principal)', size=12)
                axis.set_xticklabels(pc1_loadings['Attribute'],rotation='vertical')
                
                ax[1][1].plot(clf[1].explained_variance_ratio_.cumsum(), lw=3, color='#087E8B')
                ax[1][1].set_title('Variação Explicatória Cumulativa', size=12)
                ax[1][1].set_xlabel('Número de Componentes Principais')

            else:
                try:
                    importances = pd.DataFrame(data={
                        'Attribute': X_values.columns,
                        'Importance': clf[1].coef_[0]
                    })
                except:
                    importances = pd.DataFrame(data={
                        'Attribute': X_values.columns,
                        'Importance': clf[1].feature_importances_
                    })
                importances = importances.sort_values(by='Importance', ascending=False)

                axis.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
                axis.set_title('Feature importances from '+clf.steps[1][0], size=12)
                axis.set_xticklabels(importances['Attribute'],rotation='vertical')
            
    return loadings
    
    
 ## ---

def display_percent_count_plot(ax, feature, rd):
    """
        Exibe valor percentual de contagens para cada valor x em
        gráfico countplot seaborn. 
        
        inputs: - ax (axis on which the count plot is drawn):              matplotlib.Axis
        ------- - feature (feature displayed on the x axis):               pandas.Series
                - rd (number of digits from the decimal point to display): int        
        outputs: None
        -------
    """
    
    total = len(feature)
    for p in ax.patches:
        percentage = ("{:."+str(rd)+"f}%").format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size = 12)
        
    return None

## ---

def cf_matrix_labels(cf_matrix):
    """
        Produz rótulos a partir de uma matriz de confusão 2x2 para plotar
        em um heatmap seaborn. 
        
        inputs: - cf_matrix (2x2 confusion matrix):     numpy 2D array
        -------
        
        outputs: labels (a list containing the labels): numpy 1D array
        --------
    """
    
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    return labels
    
## ---

def dist_medians(df, feat, hue):
    """
        Calcula as diferenças entre as medianas de uma determinada variável aleatória
        considerando os níveis de matiz.
        
        inputs: - df (data) :                  pandas.DataFrame
        ------- - feat (feature name):         str
                - hue (name of hue variable ): str
                
        outputs: a pandas.Series containing the difference between medians
        --------
    """
    medians = [df[df[hue] == val][feat].median() for val in df[hue].unique()]
    return pd.Series(medians).diff()[1:]
    
## --- 

def tryconvert(value, default, *types):
    for t in types:
        try:
            return t(value)
        except (ValueError, TypeError):
            continue
    return default

