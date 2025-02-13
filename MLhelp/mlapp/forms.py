from django import forms
#########PREPROCESSING FORM###############
class PreprocessingForm(forms.Form):
    # Encoders, imputers, scalers, and vectorizers options
    ENCODER_CHOICES = [
        ('OneHotEncoder', 'One Hot Encoder'),
        ('LabelEncoder', 'Label Encoder'),
        ('OrdinalEncoder', 'Ordinal Encoder'),
        ('passthrough','No Encoder')
    ]
    
    IMPUTER_CHOICES = [
        ('mean', 'Mean Imputation'),
        ('median', 'Median Imputation'),
        ('most_frequent', 'Most Frequent'),
        #('constant', 'Constant'),
        ('passthrough','No Imputer')
    ]
    
    SCALER_CHOICES = [
        ('StandardScaler', 'Standard Scaler'),
        ('MinMaxScaler', 'Min-Max Scaler'),
        ('RobustScaler', 'Robust Scaler'),
        ('passthrough', 'No Scaling'),
    ]
    
    VECTOR_CHOICES = [
        ('CountVectorizer', 'Count Vectorizer'),
        ('TfidfVectorizer', 'TF-IDF Vectorizer'),
        ('passthrough', 'No Vectorizer'),
    ]
    
    # Transformer choices
    encoder    = forms.ChoiceField(choices=ENCODER_CHOICES, label="Encoder", required=False)
    imputer    = forms.ChoiceField(choices=IMPUTER_CHOICES, label="Imputer", required=False)
    scaler     = forms.ChoiceField(choices=SCALER_CHOICES, label="Scaler", required=False)
    vectorizer = forms.ChoiceField(choices=VECTOR_CHOICES, label="Vectorizer", required=False)

    # Parameters for custom imputation or scaling
    #constant_value = forms.FloatField(label="Constant Value (for Imputer)", required=False, min_value=0.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        column_type = self.data.get('column_type') or self.initial.get('column_type')
        
        # Dynamically adjust available fields based on column type
        if column_type == 'numeric':
            self.fields['encoder'].widget = forms.HiddenInput()  # No encoder for numeric columns
            self.fields['vectorizer'].widget = forms.HiddenInput()  # No vectorizer for numeric columns
        elif column_type == 'categorical':
            self.fields['scaler'].widget = forms.HiddenInput()  # No scaler for categorical columns
            self.fields['vectorizer'].widget = forms.HiddenInput()  # No vectorizer for categorical columns
        elif column_type == 'text':
            self.fields['encoder'].widget = forms.HiddenInput()  # No encoder for text columns
            self.fields['imputer'].widget = forms.HiddenInput()  # No imputer for text columns
            self.fields['scaler'].widget = forms.HiddenInput()  # No scaler for text columns
        elif column_type == 'datetime':
            self.fields['encoder'].widget = forms.HiddenInput()  # No encoder for datetime columns
            self.fields['imputer'].widget = forms.HiddenInput()
            self.fields['scaler'].widget = forms.HiddenInput()
            self.fields['vectorizer'].widget = forms.HiddenInput()



###########MODEL_FORM##########

class ModelTypeForm(forms.Form):
    def __init__(self, is_target=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Détermine les choix en fonction de la présence d'une cible
        if is_target == 'yes':
            model_type_choices = [
                ('classification', 'Classification'),
                ('regression', 'Régression'),
            ]
        else:
            model_type_choices = [
                ('clustering', 'Clustering'),
            ]
        self.fields['model_type'] = forms.ChoiceField(
            choices=model_type_choices,
            required=True,
            widget=forms.Select(attrs={'id': 'model-type'})
        )

CLASSIFICATION_MODEL_CHOICES = [
    ('random_forest', 'Random Forest'),
    ('svm', 'SVM'),
    ('logistic_regression', 'Régression Logistique'),
]

REGRESSION_MODEL_CHOICES = [
    ('linear_regression', 'Régression Linéaire'),
    ('ridge_regression', 'Régression Ridge'),
    ('decision_tree', 'Arbre de Décision'),
]

CLUSTERING_MODEL_CHOICES = [
    ('kmeans', 'K-Means'),
    ('dbscan', 'DBSCAN'),
    ('hierarchical', 'Clustering Hiérarchique'),
]

class ClassificationModelForm(forms.Form):
    model = forms.ChoiceField(
        label="Modèle",
        choices=CLASSIFICATION_MODEL_CHOICES,
        widget=forms.Select(attrs={'id': 'classification-model'})
    )

class RegressionModelForm(forms.Form):
    model = forms.ChoiceField(
        label="Modèle",
        choices=REGRESSION_MODEL_CHOICES,
        widget=forms.Select(attrs={'id': 'regression-model'})
    )

class ClusteringModelForm(forms.Form):
    model = forms.ChoiceField(
        label="Modèle",
        choices=CLUSTERING_MODEL_CHOICES,
        widget=forms.Select(attrs={'id': 'clustering-model'})
    )

###########CLASSIFICATION############
class RandomForestHyperparametersForm(forms.Form):
    n_estimators = forms.IntegerField(
        label="Nombre d'estimateurs",
        min_value=1,
        initial=100,
        help_text="Nombre d'arbres dans la forêt."
    )
    max_depth = forms.IntegerField(
        label="Profondeur maximale",
        min_value=1,
        required=False,
        help_text="Profondeur maximale de chaque arbre."
    )

class SVMHyperparametersForm(forms.Form):
    C = forms.FloatField(
        label="C",
        min_value=0.0,
        initial=1.0,
        help_text="Paramètre de régularisation."
    )
    kernel = forms.ChoiceField(
        label="Noyau",
        choices=[('linear', 'Linéaire'), ('rbf', 'RBF'), ('poly', 'Polynomial')],
        initial='rbf'
    )


class LogisticRegressionHyperparametersForm(forms.Form):
    C = forms.FloatField(
        label="C",
        min_value=0.0,
        initial=1.0,
        help_text="Paramètre de régularisation. Plus la valeur de C est petite, plus la régularisation est forte."
    )
    max_iter = forms.IntegerField(
        label="Nombre d'itérations",
        min_value=50,
        initial=100,
        help_text="Le nombre maximal d'itérations pour le solveur."
    )
    solver = forms.ChoiceField(
        label="Solveur",
        choices=[('liblinear', 'Liblinear'), ('saga', 'SAGA'), ('newton-cg', 'Newton-CG'), ('lbfgs', 'LBFGS')],
        initial='liblinear',
        help_text="Le type de solveur utilisé pour la régression logistique."
    )
    penalty = forms.ChoiceField(
        label="Pénalité",
        choices=[('l2', 'L2'), ('l1', 'L1'), ('elasticnet', 'ElasticNet'), ('none', 'Aucune')],
        initial='l2',
        help_text="La norme de régularisation à utiliser."
    )
    tol = forms.FloatField(
        label="Tolérance",
        min_value=1e-6,
        initial=1e-4,
        help_text="Critère d'arrêt pour les itérations du solveur. Lorsque le changement de la fonction d'objectif est inférieur à cette tolérance, l'algorithme s'arrête."
    )

###########REGRESSION############
class LinearRegressionHyperparametersForm(forms.Form):
    fit_intercept = forms.BooleanField(
        label="Inclure l'intercept",
        initial=True,
        required=False,
        help_text="Indique si le modèle doit inclure un intercept."
    )
    normalize = forms.BooleanField(
        label="Normaliser",
        initial=False,
        required=False,
        help_text="Si True, les variables d'entrée seront normalisées."
    )


class RidgeRegressionHyperparametersForm(forms.Form):
    alpha = forms.FloatField(
        label="Alpha",
        min_value=0.0,
        initial=1.0,
        help_text="Le paramètre de régularisation."
    )
    fit_intercept = forms.BooleanField(
        label="Inclure l'intercept",
        initial=True,
        required=False,
        help_text="Indique si le modèle doit inclure un intercept."
    )
    normalize = forms.BooleanField(
        label="Normaliser",
        initial=False,
        required=False,
        help_text="Si True, les variables d'entrée seront normalisées."
    )


class DecisionTreeHyperparametersForm(forms.Form):
    max_depth = forms.IntegerField(
        label="Profondeur maximale",
        min_value=1,
        required=False,
        help_text="Profondeur maximale de l'arbre de décision."
    )
    min_samples_split = forms.IntegerField(
        label="Min. échantillons pour scinder",
        min_value=2,
        initial=2,
        help_text="Le nombre minimal d'échantillons requis pour diviser un nœud interne."
    )
    min_samples_leaf = forms.IntegerField(
        label="Min. échantillons pour feuille",
        min_value=1,
        initial=1,
        help_text="Le nombre minimal d'échantillons requis dans une feuille."
    )

###########CLUSTERING############
class KMeansHyperparametersForm(forms.Form):
    n_clusters = forms.IntegerField(
        label="Nombre de clusters",
        min_value=2,
        initial=3,
        help_text="Nombre de clusters à former."
    )
    init = forms.ChoiceField(
        label="Méthode d'initialisation",
        choices=[('k-means++', 'k-means++'), ('random', 'Aléatoire')],
        initial='k-means++'
    )


class DBSCANHyperparametersForm(forms.Form):
    eps = forms.FloatField(
        label="Epsilon",
        min_value=0.0,
        initial=0.5,
        help_text="Distance maximale entre deux échantillons pour les considérer comme voisins."
    )
    min_samples = forms.IntegerField(
        label="Min. échantillons",
        min_value=1,
        initial=5,
        help_text="Le nombre minimal d'échantillons dans un voisinage pour qu'un point soit un noyau."
    )


class HierarchicalClusteringHyperparametersForm(forms.Form):
    n_clusters = forms.IntegerField(
        label="Nombre de clusters",
        min_value=2,
        initial=2,
        help_text="Le nombre de clusters que l'algorithme doit former."
    )
    linkage = forms.ChoiceField(
        label="Méthode de liaison",
        choices=[('ward', 'Ward'), ('complete', 'Complete'), ('average', 'Average')],
        initial='ward',
        help_text="La méthode utilisée pour calculer la distance entre les clusters."
    )
    affinity = forms.ChoiceField(
        label="Affinité",
        choices=[('euclidean', 'Euclidienne'), ('manhattan', 'Manhattan'), ('cosine', 'Cosinus')],
        initial='euclidean',
        help_text="La mesure de distance utilisée dans le calcul des clusters."
    )