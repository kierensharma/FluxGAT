import cobra

model = cobra.io.read_sbml_model("data/iCHO2291.xml")

model_stoichiometry = cobra.util.array.create_stoichiometric_matrix(model, array_type='DataFrame')
model_stoichiometry.to_pickle('data/iCHO2291_stoichiometry.pkl')

gene_names = []

for gene in model.genes:
    gene_names.append(gene.id)

with open('data/gene_names.txt', 'w') as f:
    for item in gene_names:
        f.write("%s\n" % item)