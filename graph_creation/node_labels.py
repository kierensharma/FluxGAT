import cobra
import pandas as pd
from collections import defaultdict
import numpy as np
import re

def convert_to_logic_string(input_string, gene_dict):
    """
    Converts a gene protein reaction (GPR) rule to a logic string with values 
    based on a dictionary mapping gene IDs to their experimental essentiality.

    Parameters:
    input_string (str): The GPR rule string.
    gene_dict (dict): A dictionary mapping gene IDs (as integers) to their essentiality values (0 or 1).

    Returns:
    str: A logic string with gene IDs replaced by their essentiality.
    """
    components = re.split(r'(\s+|\(|\))', input_string)
    converted_components = [
        str(1 - gene_dict[int(component)]) if component.isdigit() else component 
        for component in components
    ]
    return ''.join(filter(None, converted_components))

def process_logic(expression):
    """
    Evaluates a GPR rule logis string containing 'and', 'or', and parentheses.

    Parameters:
    expression (str): The logic expression string to evaluate.

    Returns:
    int: The reaction essentiality (0 or 1) from evaluating its GPR rule.
    """
    def evaluate(expr):
        if expr.isdigit():
            return int(expr)
        i = 0
        while i < len(expr):
            if expr[i] == '(':
                count = 1
                j = i + 1
                while j < len(expr) and count > 0:
                    if expr[j] == '(':
                        count += 1
                    elif expr[j] == ')':
                        count -= 1
                    j += 1
                inner_value = evaluate(expr[i + 1:j - 1])
                expr = expr[:i] + str(inner_value) + expr[j:]
                i = i + len(str(inner_value)) - 1
            i += 1
        if 'and' in expr:
            return min(evaluate(sub_expr) for sub_expr in expr.split(' and '))
        elif 'or' in expr:
            return max(evaluate(sub_expr) for sub_expr in expr.split(' or '))
        return int(expr)
    return evaluate(expression)

def reaction_essentiality_calc(model, gene_essentiality_lookup):
    """
    Calculates the essentiality of reactions in a metabolic model based on GPR rules and 
    gene essenitlaity lables.

    Parameters:
    model (cobra.Model): The metabolic model.

    Returns:
    pandas.DataFrame: A DataFrame with reaction IDs and their essentiality based on gene expressions.
    """
    reaction_values = [reaction.id for reaction in model.reactions]
    dict_df = defaultdict(list)
    for r in reaction_values:
        reaction = model.reactions.get_by_id(r)
        genes = [gene.id for gene in reaction.genes]
        GPR = str(reaction.gpr)
        dict_df['reaction'].append(r)
        dict_df['genes'].append(genes)
        dict_df['GPR'].append(GPR)
    df = pd.DataFrame(dict_df)

    results_df = defaultdict(list)
    for ind, row in df.iterrows():
        results_df['reaction'].append(row['reaction'])
        if row['GPR']:
            converted_string = convert_to_logic_string(row['GPR'], gene_essentiality_lookup)
            essentiality = 1 - process_logic(converted_string)
        else:
            essentiality = np.nan
        results_df['essentiality'].append(essentiality)

    return pd.DataFrame(results_df)

def main():
    try:
        model = cobra.io.read_sbml_model("../data/iCHO2291.xml")
        essentiality_df = pd.read_csv("../data/gene_essentiality.csv")
        gene_essentiality_lookup = dict(zip(essentiality_df['GeneID'], essentiality_df['Experimental_Essentiality']))
        reaction_essentiality_df = reaction_essentiality_calc(model, gene_essentiality_lookup)
        reaction_essentiality_df.to_pickle("../data/node_labels.pkl")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
