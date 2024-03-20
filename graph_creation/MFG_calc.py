import cobra
import numpy as np
import pandas as pd
import pickle

def stoichiometry_calc(model):
    """
    Calculate the stoichiometry matrix of a given metabolic model.

    Parameters:
    model (cobra.Model): The metabolic model to extract the stoichiometry from.

    Returns:
    pandas.DataFrame: A DataFrame representing the stoichiometric matrix of the model.
    """
    return cobra.util.array.create_stoichiometric_matrix(model, array_type='DataFrame')


def gene_names_calc(model):
    """
    Extract gene names from a metabolic model.

    Parameters:
    model (cobra.Model): The metabolic model to extract gene names from.

    Returns:
    list: A list of gene names (strings) in the model.
    """
    try:
        gene_names = [gene.id for gene in model.genes]
        return gene_names
    except AttributeError:
        raise ValueError("Invalid model format. The model does not contain genes attribute.")


def reversibility_calc(model):
    """
    Calculate the reversibility of reactions in a metabolic model.

    Parameters:
    model (cobra.Model): The metabolic model to evaluate reaction reversibility.

    Returns:
    list: A list of integers where 1 represents a reversible reaction and 0 represents an irreversible reaction.
    """
    try:
        reversibilities = [1 if reaction.reversibility else 0 for reaction in model.reactions]
        return reversibilities
    except AttributeError:
        raise ValueError("Invalid model format. The model does not contain reactions attribute.")

def MFG_calc(S, r, v):
    """
    Calculate the mass flow graph (MFG) based on stoichiometry matrix (S),
    reversibility vector (r), and flux sampling vector (v). This implementation is based on the methods 
    described in the following research paper:
    
    Beguerisse-Díaz, M., Bosque, G., Oyarzún, D. et al. Flux-dependent graphs for metabolic networks. 
    npj Syst Biol Appl 4, 32 (2018). https://doi.org/10.1038/s41540-018-0067-y

    Parameters:
    S (numpy.ndarray): Stoichiometry matrix.
    r (numpy.ndarray): Vector indicating the reversibility of reactions (1 for reversible, 0 for irreversible).
    v (numpy.ndarray): Flux sampling vector.

    Returns:
    numpy.ndarray: The result of the MFG calculation.
    """
    S = np.nan_to_num(S)

    I = np.eye(S.shape[1])
    D = np.diag(r)

    A = np.block([[I, np.zeros((S.shape[1], S.shape[1]))], 
                  [np.zeros((S.shape[1], S.shape[1])), D]])

    B = np.block([S, -S])

    S_2m = B.dot(A)
    S_plus_2m = (np.abs(S_2m) + S_2m) / 2
    S_minus_2m = (np.abs(S_2m) - S_2m) / 2

    W_plus = np.linalg.pinv(np.diag(S_plus_2m.dot(np.ones(2*S.shape[1]))))
    W_minus = np.linalg.pinv(np.diag(S_minus_2m.dot(np.ones(2*S.shape[1]))))

    abs_v = np.abs(v)
    matrix1 = (abs_v + v) / 2
    matrix2 = (abs_v - v) / 2
    v_2m = np.vstack((matrix1, matrix2)).flatten()

    V = np.diag(v_2m)
    j = S_plus_2m.dot(v_2m)
    J = np.diag(j)
    J_pinv = np.linalg.pinv(J)

    return np.dot(np.dot(np.transpose(S_plus_2m.dot(V)), J_pinv), S_minus_2m.dot(V))

def main():
    try:
        model = cobra.io.read_sbml_model("../data/iCHO2291.xml")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    S = stoichiometry_calc(model)
    r = reversibility_calc(model)

    try:
        flux_df = pd.read_pickle('../data/flux_sampling.pkl')
    except Exception as e:
        print(f"Error loading flux data: {e}")
        return
    
    flux_df = flux_df.iloc[10000:]
    flux_df.loc['Mean'] = flux_df.mean(axis=0)
    flux_df.loc['LQ'] = flux_df.quantile(0.25, axis=0)
    flux_df.loc['UQ'] = flux_df.quantile(0.75, axis=0)
    
    v = flux_df.loc['Mean'].to_numpy().astype(float)

    M = MFG_calc(S, r, v)

    try:
        with open('../data/MFG.pkl', 'wb') as f:
            pickle.dump(M, f)
    except Exception as e:
        print(f"Error saving MFG results: {e}")

if __name__ == "__main__":
    main()