from django import template
register = template.Library()

@register.filter
def convert_data_frame_to_html_table_headers(df):
    # En-têtes de colonnes avec champs modifiables pour chaque colonne
    html = "<tr>"
    for i, col in enumerate(df.columns):
        html += f'<th><input type="text" name="column_{i}" value="{col}"></th>'
    html += "</tr>"
    return html

@register.filter
def convert_data_frame_to_html_table_rows(df):
    html = ""
    for row_index, row_values in df.iterrows():
        row_html = '<tr>'  # Ouvrir le tag <tr> pour chaque ligne
        for col_index, value in enumerate(row_values):
            row_html += f'<td><input type="text" name="cell_{row_index}_{col_index}" value="{value}"></td>'
        row_html += '</tr>'  # Fermer le tag <tr>
        html += row_html
    return html