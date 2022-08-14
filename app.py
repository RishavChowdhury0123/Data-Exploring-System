import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
import streamlit as st
from cmasher import take_cmap_colors
import random
from sklearn.preprocessing import LabelEncoder
from matplotlib import rcParams
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
st.set_page_config(page_title='Data Profiler', layout='wide')
sns.set_style('whitegrid')

def configure_rcparams():
    rcParams['figure.figsize']= (10.5,5)
    rcParams['axes.linewidth']=0.3
    rcParams['lines.linewidth']=3
    rcParams['axes.labelsize']=15
    rcParams['axes.labelweight']='semibold'
    rcParams['axes.titlesize']=16
    rcParams['axes.titleweight']='semibold'
    rcParams['axes.edgecolor']='grey'
    rcParams['font.stretch']='semi-condensed'
    rcParams['font.family']='Arial'
    rcParams['legend.frameon']=False
    rcParams['legend.fontsize']=10.5
    rcParams['ytick.labelsize']=12.5
    rcParams['xtick.labelsize']=12.5
    rcParams['axes.spines.top']=False
    rcParams['axes.spines.right']=False
    rcParams['grid.linestyle']='--'
    rcParams['grid.color']='grey'
    rcParams['grid.alpha']= 0.3

def initialize_keys():
    if 'nrows' not in st.session_state:
        st.session_state['nrows']= 5

    if 'type' not in st.session_state:
        st.session_state['type']='Numerical'

    if 'selectbox' not in st.session_state:
        st.session_state['selectbox']= None 

    if 'visuals' not in st.session_state and st.session_state['type']=='Numerical':
        st.session_state['visuals']= 'Histogram'
    elif 'visuals' not in st.session_state and st.session_state['type']!='Numerical':
        st.session_state['visuals']= 'Count Bars'

    if 'bivariate_kind' not in st.session_state:
        st.session_state['bivariate_kind']=None


    if 'multivariate_kind' not in st.session_state:
        st.session_state['multivariate_kind']=None

    if 'coef_method' not in st.session_state: 
        st.session_state['coef_method']="Pearson's"

    if 'model_kind' not in st.session_state:
        st.session_state['model_kind']= 'Regression'

    if 'target' not in st.session_state:
        st.session_state['target']= None

    if 'select_model' not in st.session_state:
        st.session_state['select_model']= None
    
    if 'model_plot_kind' not in st.session_state:
        st.session_state['model_plot_kind']= None
    
    if 'log' not in st.session_state:
        st.session_state['log']=False

    if 'scale' not in st.session_state:
        st.session_state['scale']=False

configure_rcparams()
initialize_keys()

# To transform numbers to abbreviated format
def format_numbers(number, pos=None, fmt= '.0f'):
    fmt= '%'+fmt
    thousands, millions, billions= 1_000, 1_000_000, 1_000_000_000
    if number/billions >=1:
        return (fmt+'B') %(number/billions)
    elif number/millions >=1:
        return (fmt+'M') %(number/millions)
    elif number/thousands >=1:
        return (fmt+'K') %(number/thousands)
    else:
        return fmt %(number)

def min_max(x):
    return [(i-x.min())/(x.max()-x.min()) for i in x]

# Function for getting color codes
def get_color_codes(name, size):
  return take_cmap_colors(name, size, return_fmt='hex')

# Function for random colors codes
def get_random_colors(n=1):
  if n==1:
    return get_color_codes(name= random.choice(plt.colormaps()), size=n)[0]
  else:
    return get_color_codes(name= random.choice(plt.colormaps()), size= n)

@st.experimental_memo
def load_data(uploader):
    if uploader is not None:
        try:
            return pd.read_csv(uploader)
        except:
            pass

def show_general_overview(df):
    with st.expander(label='General Overview'):
        st.markdown(f'''
                        <p><b>No of records: </b> {format_numbers(df.shape[0], fmt='.1f')}<p>''',unsafe_allow_html=True)
        st.markdown(f'''
                        <p><b>No of columns: </b> {df.shape[1]}<p>''',unsafe_allow_html=True)
        st.markdown(f'''
                        <p><b>No of duplicate records: </b> {format_numbers(df.duplicated().sum(), fmt='.1f')}<p>''',unsafe_allow_html=True)
        missing_vals= (df.shape[0]-df.dropna().shape[0])
        if missing_vals == 0:
            st.markdown(f'''
                <p><b>No of records with missing values: </b> None <p>''',unsafe_allow_html=True)
        else: 
            st.markdown(f'''
                        <p><b>No of records with missing values: </b> {format_numbers(missing_vals, fmt='.1f')}<p>''',unsafe_allow_html=True)
        
        all_cols= ', '.join(df.columns.tolist())
        st.markdown(f'<p><b>Feature Name(s): </b> {all_cols}</p>', unsafe_allow_html=True)

        int_cols= ', '.join(df.select_dtypes('int64').columns.tolist())
        if len(int_cols) > 0:
            st.markdown(f'<p><b>Integer Feature(s): </b> {int_cols}</p>', unsafe_allow_html=True)

        float_cols= ', '.join(df.select_dtypes('float64').columns.tolist())
        if len(float_cols) > 0:
            st.markdown(f'<p><b>Float Feature(s): </b> {float_cols}</p>', unsafe_allow_html=True)

        cat_cols= ', '.join(df.select_dtypes('O').columns.tolist())
        if len(cat_cols) > 0:
            st.markdown(f'<p><b>Object Feature(s): </b> {cat_cols}</p>', unsafe_allow_html=True)

        st.markdown(f'''
                        <p><b>Memory Usage: </b> {df.memory_usage(deep=True).sum()/1024**2:.2f} MiBs<p>''',unsafe_allow_html=True)

def show_descriptive_stats(df):
    with st.expander('Descriptive Statistics'):
        st.write(df.describe().iloc[1:,:].applymap(lambda x: '{:.2f}'.format(x)).T)

def univariate_plotter(df, col, datatype):
    visual_dicts= {'Quantitative':['Histogram','Density Plot','Box Plot'], 'Qualitative': ['Count Bars','Pie Chart']}
    st_cols= st.columns(2)
    st.session_state['visuals']=st_cols[1].radio('Choose kind of visualisation', options=visual_dicts.get(datatype))
    if datatype== 'Quantitative':
        st.session_state['log']= st_cols[1].checkbox(label='Logarithmic')
        st_cols[0].markdown(f'<b>Mean:</b> {df[col].mean():.3f}', unsafe_allow_html= True)
        st_cols[0].markdown(f'<b>Standard Deviation:</b> {df[col].std():.3f}',  unsafe_allow_html= True)
        st_cols[0].markdown(f'<b>Minimum:</b> {df[col].min()}',  unsafe_allow_html= True)
        st_cols[0].markdown(f'<b>Maximum:</b> {df[col].max()}',  unsafe_allow_html= True)
        fig, ax= plt.subplots(figsize= (8,5))
        if st.session_state['log']==False:
            if st.session_state['visuals']=='Histogram':
                n, bins, patches= ax.hist(df[col], bins=20, alpha=0.85, edgecolor= 'white')
                cmap= plt.cm.get_cmap('Oranges_r')
                for rect,c in zip(patches, cmap(min_max(bins))):
                    rect.set_facecolor(c)
                norm= plt.Normalize(vmin= min(bins), vmax= max(bins))
                sm= plt.cm.ScalarMappable(norm= norm, cmap= cmap)
                ax.set_title(col)
                plt.legend()
            elif st.session_state['visuals']=='Density Plot':
                sns.kdeplot(df[col], ax= ax)
                ax.set_title(col)
            else:
                ax.boxplot(x= df[col])
                ax.set_title(col)
            st_cols[1].pyplot(fig)
        else:
            if st.session_state['visuals']=='Histogram':
                n, bins, patches= ax.hist(np.log(df[col]), bins=20, alpha=0.85, edgecolor= 'white')
                cmap= plt.cm.get_cmap('Oranges_r')
                for rect,c in zip(patches, cmap(min_max(bins))):
                    rect.set_facecolor(c)
                norm= plt.Normalize(vmin= min(bins), vmax= max(bins))
                sm= plt.cm.ScalarMappable(norm= norm, cmap= cmap)
                ax.set_title(col)
                plt.legend()
            elif st.session_state['visuals']=='Density Plot':
                sns.kdeplot(np.log(df[col]), ax= ax)
                ax.set_title(col)
            else:
                ax.boxplot(x= np.log(df[col]))
                ax.set_title(col)
            st_cols[1].pyplot(fig)
    else:
        st_cols[0].markdown(f'<p><b>No of unique value(s)</b>:{df[col].nunique()}</p>', unsafe_allow_html=True)
        st_cols[0].markdown(f'<p><b>Unique value(s)</b>:{df[col].unique().tolist()}</p>', unsafe_allow_html=True)
        st_cols[0].markdown(f'<p><b>Most frequent </b>:{df[col].value_counts().index[0]}</p>', unsafe_allow_html=True)
        fig= plt.figure(figsize= (8,5))
        if st.session_state['visuals']=='Count Bars':
            sns.countplot(y=df[col], order= df[col].value_counts().index, palette='cividis')
        elif st.session_state['visuals']== 'Pie Chart':
            explode= [0.05,]*(df[col].nunique())
            plt.pie(df[col].value_counts(), labels= df[col].value_counts().index,
            explode= explode, autopct= '%.1f%%', colors= get_color_codes('OrRd_r', df[col].nunique()))
        st_cols[1].pyplot(fig)

def bivariate_plotter(df, col1, col2, quant_cols, o_cols):
    visual_dicts= {'num_to_num':['Scatter Plot'], 'num_to_cat': ['Bars','Box Plot','Strip Plot'],
                    'cat_to_cat':['Stacked Bars','Unstacked Bars']}
    st_cols= st.columns(2)
    if col1 in o_cols and col2 in o_cols:
        st.session_state['bivariate_kind']= st_cols[1].radio('Choose kind of visualisation', visual_dicts.get('cat_to_cat'))
    elif col1 in o_cols and col2 in quant_cols:
        st.session_state['bivariate_kind']= st_cols[1].radio('Choose kind of visualisation', visual_dicts.get('num_to_cat'))
    elif col1 in quant_cols and col2 in o_cols:
        st.session_state['bivariate_kind']= st_cols[1].radio('Choose kind of visualisation', visual_dicts.get('num_to_cat'))
    else:
        st.session_state['bivariate_kind']= st_cols[1].radio('Choose kind of visualisation', visual_dicts.get('num_to_num'))

    st_cols[0].markdown(f"Comparing <b>'{col1}'</b> to <b>'{col2}'</b>", unsafe_allow_html=True)
    if col1 in quant_cols and col2 in quant_cols:
        st_cols[0].write("Pearson's Correlation Coefficient: {:.3f}".format(np.corrcoef(x= df[col1],y= df[col2])[0,1]))
    elif col1 in o_cols and col2 in quant_cols:
        st_cols[0].write(df.groupby(col1)[col2].describe().applymap(lambda x: '{:.2f}'.format(x))[['mean','std','min','max']])
    elif col1 in quant_cols and col2 in o_cols:
        st_cols[0].write(df.groupby(col2)[col1].describe().applymap(lambda x: '{:.2f}'.format(x))[['mean','std','min','max']])
    elif col1 in o_cols and col2 in o_cols:
        st_cols[0].write(pd.crosstab(df[col1], df[col2]))

    fig, ax= plt.subplots(figsize= (8,5))
    if st.session_state['bivariate_kind']=='Stacked Bars':
        df.groupby(col1)[col2].value_counts().unstack().plot(kind='barh', stacked=True, ax= ax, color= get_color_codes('mako', df[col2].nunique()))
        ax.set(**{'xlabel':col2, 'ylabel': col1, 'title': f'{col2} vs {col1}'})
    elif st.session_state['bivariate_kind']=='Unstacked Bars':
        pd.crosstab(index= df[col1], columns= df[col2]).plot(kind='barh', ax= ax, color= get_color_codes('mako', df[col2].nunique()))
        ax.set(**{'xlabel':col2, 'ylabel': col1, 'title': f'{col2} vs {col1}'})
    elif st.session_state['bivariate_kind']=='Bars':
        if col1 in o_cols:
            df.groupby(col1)[col2].mean().plot(kind='barh',ax= ax, color= get_color_codes('mako',df[col1].nunique()))
            ax.set(**{'xlabel':col2, 'ylabel': col1, 'title': f'{col2} vs {col1}'})
        else:
            df.groupby(col2)[col1].mean().plot(kind='barh', ax= ax, color= get_color_codes('mako', df[col2].nunique()))
            ax.set(**{'xlabel':col1, 'ylabel': col2, 'title': f'{col1} vs {col2}'})
    elif st.session_state['bivariate_kind']=='Box Plot':
        if col1 in o_cols:
            sns.boxplot(df[col1], df[col2], ax= ax, palette='YlGnBu')
            ax.set(**{'xlabel':col1, 'ylabel': col2})
        else:
            sns.boxplot(df[col2], df[col1], ax= ax, palette='YlGnBu')
            ax.set(**{'xlabel':col2, 'ylabel': col1, 'title': f'{col2} vs {col1}'})
    elif st.session_state['bivariate_kind']=='Strip Plot':
        if col1 in o_cols:
            sns.stripplot(df[col1], df[col2], ax= ax, palette='cividis')
            ax.set(**{'xlabel':col1, 'ylabel': col2, 'title': f'{col1} vs {col2}'})
        else:
            sns.stripplot(df[col2], df[col1], ax= ax, palette='viridis')
            ax.set(**{'xlabel':col2, 'ylabel': col1, 'title': f'{col2} vs {col1}'})
    else:
        ax.scatter(df[col1], df[col2], alpha=0.5,cmap='winter')
        ax.set(**{'xlabel':col1, 'ylabel': col2, 'title': f'{col1} vs {col2}'})
    st_cols[1].pyplot(fig)

def multivariate_plotter(df, col1, col2, col3, quant_cols, o_cols):
    st_cols= st.columns(2)
    st_cols[0].markdown(f"Comparing <b>'{col1}'</b> to <b>'{col2}'</b> vs <b>'{col3}'</b>", unsafe_allow_html=True)
    if col1 in quant_cols and col2 in quant_cols:
        if col3 in o_cols:
            corrs= df.groupby(col3).agg({col1: 'mean', col2:'mean'})
        else:
            corrs= df[[col1, col2, col3]].corr()
        st_cols[0].write("Pearson's Correlation Coefficients")
        st_cols[0].write(corrs)
    elif col1 in o_cols and col2 in quant_cols:
        if col3 in o_cols:
            ctab= df.groupby([col1, col3])[col2].mean().unstack()
        else:
            ctab= df.groupby(col1).agg({col2: 'mean', col3:'mean'})
        st_cols[0].write(ctab)
    elif col1 in quant_cols and col2 in o_cols:
        if col3 in o_cols:
            ctab= df.groupby([col2, col3])[col1].mean().unstack()
        else:
            ctab= df.groupby(col2).agg({col1: 'mean', col3:'mean'})
        st_cols[0].write(ctab)
    elif col1 in o_cols and col2 in o_cols:
        if col3 in o_cols:
            ctab= df.groupby([col1, col2])[col3].value_counts().unstack().reset_index()
        else:
            ctab= pd.crosstab(index=df[col1],columns= df[col2],values= df[col3], aggfunc='mean')
        st_cols[0].write(ctab)

    fig, ax= plt.subplots(figsize=(8,5))
    if col1 in o_cols and col2 in o_cols:
        if col3 in o_cols:
            ctab= df.groupby([col1, col2])[col3].value_counts().unstack()
            ctab.plot(kind='barh', ax= ax, color= get_color_codes('cool',df[col1].nunique()))
        else:
            ctab= df.groupby([col1, col2])[col3].mean().unstack()
            ctab.plot(kind='barh', ax= ax, color= get_color_codes('autumn',df[col2].nunique()), alpha=0.75)
        
        ax.set(**{'ylabel':col1, 'xlabel': col3,'title':f'{col1} vs {col3}'})
    elif col1 in o_cols and col2 in quant_cols:
        if col3 in o_cols:
            ctab= df.groupby([col1,col3])[col2].mean().unstack()
            ctab.plot(kind='barh', ax= ax, color= get_color_codes('turbo',df[col3].nunique()))
            ax.set(**{'xlabel':col2, 'ylabel': col1,'title':f'{col2} vs {col1}'})
        else:
            sns.scatterplot(df[col2], df[col3], hue= df[col1], size= df[col3], ax= ax, sizes= (50,170), palette='coolwarm')
            ax.set(**{'xlabel':col2, 'ylabel': col3,'title':f'{col2} vs {col3}'})
    elif col1 in quant_cols and col2 in o_cols:
        if col3 in o_cols:
            ctab= df.groupby([col2,col3])[col1].mean().unstack()
            ctab.plot(kind='barh', ax= ax, color= get_color_codes('viridis',df[col3].nunique()))
            ax.set(**{'xlabel':col1, 'ylabel': col2, 'title': f'{col1} vs {col2}'})
        else:
            sns.scatterplot(df[col1], df[col3], hue= df[col2], size= df[col2], ax= ax, sizes= (70,180), alpha=0.7, palette='YlGnBu')
            ax.set(**{'xlabel':col1, 'ylabel': col3,'title': f'{col1} vs {col3}' })
    elif col1 in quant_cols and col2 in quant_cols:
        if col3 in o_cols:
            sns.scatterplot(df[col1], df[col2], hue= df[col3], size= df[col3], ax= ax, sizes= (70,180), alpha=0.7, palette='Greens')
            ax.set(**{'xlabel':col1, 'ylabel': col2, 'title': f'{col1} vs {col2}'})
        else:
            sns.scatterplot(df[col1], df[col2], hue= df[col3], size= df[col3], ax= ax, sizes= (70,180), alpha=0.7, palette='viridis')
            ax.set(**{'xlabel':col1, 'ylabel': col2, 'title': f'{col1} vs {col2}'})
    st_cols[1].pyplot(fig)
    fig, ax= plt.subplots(figsize= (8,8))
    dummy= df.copy()
    dummy[dummy.select_dtypes('O').columns]= dummy.select_dtypes('O').apply(LabelEncoder().fit_transform)
    st.write('Correlation')
    st.session_state['coef_method']= st.radio('Choose Method', ["Pearson's", "Spearman's"])
    if st.session_state['coef_method']=="Pearson's": sns.heatmap(dummy.corr(method='pearson'), ax= ax, annot=True)
    else: sns.heatmap(dummy.corr(method='spearman'), ax= ax, cmap='YlGnBu',annot=True)
    st.pyplot(fig)

def show_univariates(df, quant_cols, o_cols):
    columns_dict= { 'Quantitative': quant_cols, 
                'Qualitative': o_cols}
    with st.expander('Univariate Analytics'):
        st_cols= st.columns(5)
        for col in st_cols[2:]: col.empty()
        st.session_state['type']= st_cols[0].radio('Type', ['Quantitative','Qualitative'])
        st.session_state['selectbox']= st_cols[1].selectbox(label='Choose a column', options= columns_dict.get(st.session_state['type']))
        col= st.session_state['selectbox']
        univariate_plotter(df, col= st.session_state['selectbox'], datatype=st.session_state['type'])

def show_bivariates(df, quant_cols, o_cols):
    with st.expander('Bivariate Analytics'):
        cols= st.columns(2)
        st.session_state['Variable X']= cols[0].selectbox('Variable X', df.columns)
        st.session_state['Variable Y']= cols[1].selectbox('Variable Y', df.columns.drop(st.session_state['Variable X']))
        col1, col2= st.session_state['Variable X'], st.session_state['Variable Y']
        bivariate_plotter(df= df, col1= col1, col2= col2, quant_cols=quant_cols, o_cols= o_cols)

def show_multivariates(df, quant_cols, o_cols):
    with st.expander('Multivariate Analytics'):
        cols= st.columns(3)
        var_xx= cols[0].selectbox('Variable X', df.columns, key= 'var_xx')
        var_yy= cols[1].selectbox('Variable Y', df.columns.drop(st.session_state['var_xx']), key= 'var_yy')
        var_zz= cols[2].selectbox('Variable Z', df.columns.drop(st.session_state['var_xx']).drop(st.session_state['var_yy']), key= 'var_zz')
        col1, col2, col3= st.session_state['var_xx'], st.session_state['var_yy'], st.session_state['var_zz']
        multivariate_plotter(df= df, col1= col1, col2= col2, col3= col3, quant_cols= quant_cols, o_cols= o_cols)

def model(X_train, y_train, X_test, y_test, kind, model_dict):
    from sklearn.model_selection import cross_validate
    from sklearn.utils import shuffle
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    cv_results= pd.DataFrame()
    for model_name, est in model_dict.items():
        if kind is 'reg':
            metrics= ('r2', 'neg_mean_squared_error')
        else:    
            metrics= ('accuracy', 'f1')
        
        pipe= Pipeline(steps = [('sc', StandardScaler()), ('model',est)])
        results= pd.DataFrame(cross_validate(estimator=pipe, X= X_train, y=y_train, scoring= metrics, cv=3, n_jobs=-1, return_train_score=True, return_estimator=False, error_score='raise')).drop('score_time', axis=1)
        if kind is 'reg':
            results.columns= ['Fit Time','Test R2','Train R2','Test MSE','Train MSE']
            results[['Train MSE','Test MSE']]= abs(results[['Train MSE','Test MSE']])
        else:
            results.columns= ['Fit Time','Test Accuracy','Train Accuracy','Test F1','Train F1']
        cols= ['Model']
        cols.extend(results.columns)
        results['Model']= model_name
        cv_results= cv_results.append(results, ignore_index=True)

    return shuffle(cv_results).reset_index(drop=True)[cols]
    
def plot_model_results(results, kind):
    st_cols= st.columns(2)
    st_cols[0].write(results)
    st.session_state['model_plot_kind']= st_cols[1].selectbox('Choose Plot', ['Line chart','Box plot','Bars'])
    fig, ax= plt.subplots(figsize= (8,5))
    if kind is 'reg':
        if st.session_state['model_plot_kind'] == 'Line chart':
            results.groupby('Model')['Train R2','Test R2'].mean().sort_values(by= ['Train R2','Test R2']).plot(ax= ax, marker='o')
        elif st.session_state['model_plot_kind'] == 'Box plot':
            sns.boxplot(results.Model, results['Test R2'], ax=ax)
        else:
            results.groupby('Model')['Train R2','Test R2'].mean().sort_values(by= ['Train R2','Test R2']).plot(kind='bar', ax= ax)
    else:
        if st.session_state['model_plot_kind'] == 'Line chart':
            results.groupby('Model')['Train F1','Test F1'].mean().sort_values(by= ['Train F1','Test F1']).plot(ax= ax, marker='o')
        elif st.session_state['model_plot_kind'] == 'Box plot':
            sns.boxplot(results.Model, results['Test F1'], ax=ax)
        else:
            results.groupby('Model')['Train F1','Test F1'].mean().sort_values(by= ['Train F1','Test F1']).plot(kind='bar', ax= ax)
    plt.xticks(rotation=90)
    st_cols[1].pyplot(fig)

def return_selected_model_dict(selected_models, kind, all_model_dict):
    names, models= [],[]
    for name, model in all_model_dict.get(kind).items():
        if name in selected_models:
            names.append(name)
            models.append(model)
    return dict(zip(names,models))

def show_model_selection(df):
    with st.expander('Predictive Analytics'):
        kind_dict= {'Classification': 'clf', 'Regression':'reg'}  
        all_model_dict= {'reg': {'Linear Regression': LinearRegression(), 'Lasso (L1 Regularization)': Lasso(alpha=0.05),
                            'Ridge (L2 Regularization)': Ridge(alpha=0.05), 'Decision Tree': DecisionTreeRegressor(),
                            'K-nearest Neighbors': KNeighborsRegressor(), 'Ada Boost':AdaBoostRegressor(), 'SVM':SVR(),
                            'Gradient Boost': GradientBoostingRegressor(),'Random Forest':RandomForestRegressor(),
                            'Cat Boost': CatBoostRegressor(silent=True), 'Light GBM': LGBMRegressor(), 'XG Boost': XGBRegressor(silent=True)},
                    'clf': {'Logistic Regression': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(),
                            'K-nearest Neighbors': KNeighborsClassifier(), 'Ada Boost':AdaBoostClassifier(), 'SVM': SVC(),
                            'Gradient Boost': GradientBoostingClassifier(),'Random Forest':RandomForestClassifier(),
                            'Cat Boost': CatBoostClassifier(silent=True),'Light GBM': LGBMClassifier(silent=True), 'XG Boost': XGBClassifier(silent=True)}}
        cols= st.columns(4)
        st.session_state['model_kind']=cols[0].radio('', options=kind_dict.keys())
        # st.session_state['select_model']= cols[1].multiselect('Choose learning algorithms', all_model_dict.get(kind_dict.get(st.session_state['model_kind'])).keys())
        st.session_state['target']=cols[2].selectbox('Choose Target Variable', options=df.columns)
        if kind_dict.get(st.session_state['model_kind']) is 'reg':
            st.session_state['scale']= cols[3].checkbox('Scale Target Variable')
        # model_dict= return_selected_model_dict(selected_models= st.session_state['select_model'], kind= kind_dict.get(st.session_state['model_kind']), all_model_dict=all_model_dict)
        X= df.drop(st.session_state['target'], axis=1)
        y= df[st.session_state['target']]
        if kind_dict.get(st.session_state['model_kind']) is 'clf':
            if y.nunique()==2:
                if y.dtype=='O':
                    y= LabelEncoder().fit_transform(y.values.reshape(-1,1))
                X[X.select_dtypes('O').columns]= X.select_dtypes('O').apply(LabelEncoder().fit_transform)
                X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=42, test_size=.20)
                results= model(X_train, y_train, X_test, y_test, kind=kind_dict.get(st.session_state['model_kind']), model_dict=all_model_dict.get(kind_dict.get(st.session_state['model_kind'])))
                plot_model_results(results, kind= kind_dict.get(st.session_state['model_kind']))
            else:
                st.warning('Please choose the appropriate target variable and try again.')
        else:
            if st.session_state['scale']:
                y= StandardScaler().fit_transform(y.values.reshape(-1,1))
            X[X.select_dtypes('O').columns]= X.select_dtypes('O').apply(LabelEncoder().fit_transform)
            X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=42, test_size=.20)
            results= model(X_train, y_train, X_test, y_test, kind=kind_dict.get(st.session_state['model_kind']), model_dict=all_model_dict.get(kind_dict.get(st.session_state['model_kind'])))
            plot_model_results(results, kind= kind_dict.get(st.session_state['model_kind']))
    
def actions():
    with st.sidebar:
        uploader= st.file_uploader(label= 'Upload .CSV file', accept_multiple_files=False)
    with open('./style.css') as css:
        html= "<style>{}</style>".format(css.read())

    st.markdown(html,unsafe_allow_html=True)
    df= load_data(uploader)  
    st.text('Data Explorer')
    if df is not None:
        with st.expander(f"Quick look at the data"):
            # Show more rows or less rows
            btn_cols= st.columns(7)
            up_btn= btn_cols[2].button('Show more 🔼')
            dwn_btn= btn_cols[3].button('Show less 🔽')
            for col in btn_cols[:3]:
                col.empty()
            for col in btn_cols[5:]:
                col.empty()
            if up_btn:
                st.session_state['nrows']+=5
            if dwn_btn:
                st.session_state['nrows']-=5
            
            if st.session_state['nrows']<=0:
                st.session_state['nrows']=5
            
            # Show table
            st.write(df.sample(st.session_state['nrows']))
        
        o_cols= df.select_dtypes('O').columns.to_list()
        n_cols= df.select_dtypes(exclude= 'O').columns
        i_cols= n_cols[df.select_dtypes(exclude='O').apply('nunique') <=15]
        quant_cols= list(set(n_cols).difference(set(i_cols)))
        o_cols.extend(i_cols)

        
        # general overview
        show_general_overview(df)

        # descriptive statistics
        show_descriptive_stats(df)

        # columnwise details
        try:
            show_univariates(df, quant_cols=quant_cols, o_cols=o_cols)
        except KeyError:
            pass
            
        try:
            show_bivariates(df,quant_cols=quant_cols, o_cols=o_cols)
        except:
            pass

        try:
            show_multivariates(df, quant_cols=quant_cols, o_cols=o_cols)
        except:
            pass
        # show_model_selection(df.dropna())
        try:
            show_model_selection(df.dropna())
        except:
            pass        

if __name__== '__main__':
    actions()








        
    