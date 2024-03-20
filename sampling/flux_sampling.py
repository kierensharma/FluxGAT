import cobra
from cobra.sampling import sample
import os

def sample_fluxes_and_save(model_path, output_path, num_samples=50000, thinning=1000):
    """
    Samples fluxes from a metabolic model and saves the results to a file.

    Parameters:
    model_path (str): Path to the SBML model file.
    output_path (str): Path where the sampled fluxes will be saved as a pickle file.
    num_samples (int, optional): Number of samples to generate. Defaults to 50,000.
    thinning (int, optional): Thinning factor for the sampling process. Defaults to 1,000.

    Raises:
    FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at {model_path} was not found.")

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        model = cobra.io.read_sbml_model(model_path)

        samples = sample(model, num_samples, thinning=thinning)

        samples.to_pickle(output_path)
        print(f"Flux sampling completed and saved to {output_path}")

    except Exception as e:
        print(f"An error occurred during flux sampling or file saving: {e}")

if __name__ == "__main__":
    model_file_path = "../data/iCHO2291.xml"
    output_file_path = "../data/flux_sampling.pkl"
    sample_fluxes_and_save(model_file_path, output_file_path)
