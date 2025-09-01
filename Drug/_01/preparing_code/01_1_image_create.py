from rdkit import Chem
from rdkit.Chem import Draw

mol = Chem.MolFromSmiles("CC(=O)C1=C(NC2CCCc3ccccc23)Nc4c(cccc4c5ccccc5)C1=O")
img = Draw.MolToImage(mol)
img.save("mol.png")