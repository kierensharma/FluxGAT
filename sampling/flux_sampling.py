import cobra
from cobra.sampling import sample

model = cobra.io.read_sbml_model("iCHO2291.xml")

s = sample(model, 50000, thinning=1000)