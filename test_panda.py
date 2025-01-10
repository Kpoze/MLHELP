import pandas as pd
import chardet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = "C://Users//RED94//Desktop//CSVTest.csv"

with open(dataset, 'rb') as f:
    result = chardet.detect(f.read(10000))  # Analyser les premiers 10k octets
    encoding = result['encoding']

df = pd.read_csv(dataset,encoding=encoding)
print(df)
# Demander des entrées à l'utilisateur
value_input = input("Choose the value: ")
relation_input = input("Choose the relation (==, <, >, <=, >=, !=): ")
column_input = input("Choose a column: ")

# Dictionnaire des opérateurs
operators = {
    '==': lambda x, y: x == y,
    '<': lambda x, y: x < y,
    '>': lambda x, y: x > y,
    '<=': lambda x, y: x <= y,
    '>=': lambda x, y: x >= y,
    '!=': lambda x, y: x != y,
}

# Vérifier si l'opérateur est valide
if relation_input in operators:
    # Appliquer le filtre avec l'opérateur choisi
    result = df[operators[relation_input](df[column_input], value_input)]
    print(result)
else:
    print("Opérateur non valide.")

"""column_type = df[df.columns[1]].dtype
print(column_type)
cell_type = type(df.loc[2][df.columns[2]])
print(column_type)
print(df.columns[2])
print(type(df.loc[2][df.columns[2]]))

value = "henry"
new_value = np.array(value).astype(column_type)
print(new_value)
value_type = type(new_value)
print(value_type)
if value_type == column_type : 
    print('OK')"""




"""
# Get user input
value_input  = input("Choose the value: ")
row_input    = int(input("Choose a row: "))
column_input = int(input("Choose a column: "))

# Get the column data type
column_type = df[df.columns[column_input]].dtype
print(f"Column data type: {column_type}")

# Print the current value at the location
print(f"Current value at ({row_input}, {column_input}): {df.iat[row_input, column_input]}")

try:
    # If the column type is 'object', treat value as a string (or leave it as is)
    if column_type == 'object':
        # For 'object' type, directly assign the input
        converted_value = value_input
        print(f"Converted value: {converted_value} (Type: {type(converted_value)})")
    else:
        # For other types, attempt to convert the input
        converted_value = np.array([value_input]).astype(column_type)[0]
        print(f"Converted value: {converted_value} (Type: {type(converted_value)})")
    
    # Check if the converted value type matches the column's dtype
    if column_type == 'object' or np.issubdtype(type(converted_value), column_type):
        # Update the DataFrame with the value using .iat
        df.iat[row_input, column_input] = converted_value
        print("Value successfully updated!")
    else:
        print("The value type does not match the column type. No update was made.")
except ValueError as e:
    print(f"Error converting value: {e}. No update was made.")

print("Updated DataFrame:")
print(df)
# Sauvegarder les modifications dans le CSV
df.to_csv(dataset, index=False)

# Fonction pour générer un graphique en fonction des choix de l'utilisateur
def generer_graphique(df):
    print("Types de graphiques disponibles:")
    print("1. Histogramme")
    print("2. Diagramme à barres")
    print("3. Nuage de points (Scatter plot)")
    print("4. Courbe linéaire (Line plot)")
    print("5. Boîte à moustaches (Box plot)")
    print("6. Carte de chaleur (Heatmap)")
    
    choix_graphique = input("Choisissez le type de graphique (1-6): ")
    
    # Lister les colonnes disponibles dans le DataFrame
    print("\nColonnes disponibles:", list(df.columns))
    col_x = input("Choisissez la colonne pour l'axe X: ")
    
    # Pour les graphiques nécessitant une colonne Y ou des paires de colonnes
    if choix_graphique in ['3', '4', '5']:  # Scatter plot, Line plot, Box plot
        col_y = input("Choisissez la colonne pour l'axe Y: ")
    else:
        col_y = None

    # Générer le graphique en fonction du choix
    plt.figure(figsize=(10, 6))
    
    if choix_graphique == '1':  # Histogramme
        sns.histplot(df[col_x], kde=True)
        plt.title(f"Histogramme de {col_x}")
        plt.xlabel(col_x)
        plt.ylabel("Fréquence")
    
    elif choix_graphique == '2':  # Diagramme à barres
        sns.countplot(x=col_x, data=df)
        plt.title(f"Diagramme à barres de {col_x}")
        plt.xlabel(col_x)
        plt.ylabel("Nombre")
    
    elif choix_graphique == '3':  # Nuage de points (Scatter plot)
        sns.scatterplot(x=col_x, y=col_y, data=df)
        plt.title(f"Nuage de points de {col_x} vs {col_y}")
        plt.xlabel(col_x)
        plt.ylabel(col_y)
    
    elif choix_graphique == '4':  # Courbe linéaire (Line plot)
        sns.lineplot(x=col_x, y=col_y, data=df)
        plt.title(f"Courbe linéaire de {col_x} vs {col_y}")
        plt.xlabel(col_x)
        plt.ylabel(col_y)
    
    elif choix_graphique == '5':  # Boîte à moustaches (Box plot)
        sns.boxplot(x=col_x, y=col_y, data=df)
        plt.title(f"Boîte à moustaches de {col_x} vs {col_y}")
        plt.xlabel(col_x)
        plt.ylabel(col_y)
    
    elif choix_graphique == '6':  # Carte de chaleur (Heatmap)
        corr_matrix = df.corr()  # Calcul de la matrice de corrélation
        sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
        plt.title("Carte de chaleur des corrélations")
    
    else:
        print("Choix invalide, veuillez réessayer.")
        return
    
    plt.show()

# Appel de la fonction pour générer le graphique
generer_graphique(df)"""