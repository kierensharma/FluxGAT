import cobra
import pandas as pd
from collections import defaultdict

def reaction_names_calc(model):
    """
    Extracts reaction IDs from a given metabolic model.

    Parameters:
    model (cobra.Model): The metabolic model to extract reaction IDs from.

    Returns:
    list: A list of reaction IDs (strings).
    """
    return [reaction.id for reaction in model.reactions]

def calc_node_features(model, reaction_values):
    """
    Calculates node features for each reaction in the metabolic model, including
    reactants and products.

    Parameters:
    model (cobra.Model): The metabolic model.
    reaction_values (list): A list of reaction IDs to calculate features for.

    Returns:
    pandas.DataFrame: A DataFrame containing node features for each reaction.
    """
    dict_df = defaultdict(list)

    for r in reaction_values:
        reaction = model.reactions.get_by_id(r)
        reactants_list = [reactant.id for reactant in reaction.reactants]
        products_list = [product.id for product in reaction.products]

        dict_df['reaction'].append(r)
        dict_df['reactants_and_products'].append(reactants_list + products_list)

    df = pd.DataFrame(dict_df)
    unique_reactants = sorted(set().union(*df['reactants_and_products']))
    df['reactants_one_hot'] = df['reactants_and_products'].apply(
        lambda genes: [1 if gene in genes else 0 for gene in unique_reactants])

    return df

def main():
    try:
        # Load the metabolic model
        model = cobra.io.read_sbml_model("../data/iCHO2291.xml")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    reaction_values = reaction_names_calc(model)

    node_features = calc_node_features(model, reaction_values)
    node_features.to_pickle("../data/node_features.pkl")

if __name__ == "__main__":
    main()