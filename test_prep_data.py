import unittest
from prep_data import *

def write_list_to_file(filename: str, list):
    with open("test_results/" + filename + ".txt", 'w') as f:
        for line in list:
            f.write("%s\n" % line)

class TestPrepData(unittest.TestCase):
    dna_base_smiles = {'A': 'OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1',
            'T': 'OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1',
            'G': 'OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1',
            'C': 'OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1'}
    barcode_dict_model = {
            0: 'AAAAA',
            # 1: 'AAAAC'
        }
    ab_feats_medians_model = [108.901413]
    


    def test_get_n_hydro(self):
        # data_prep = DataPrepper(self.barcode_dict_model, self.ab_feats_medians_model, self.dna_base_smiles)
        result = get_n_hydro(get_smiles_string('AAAAA'))
        print ("test_get_n_hydro")
        print (result)
        pass

    def test_get_compound_graph(self):
        smiles = get_smiles_string('AAAAA')
        resultA, resultX, = get_compound_graph(smiles, ['C', 'N', 'O', 'P'])
        write_list_to_file("test_get_compound_graph_resultA", resultA)
        write_list_to_file('test_get_compound_graph_resultX', resultX)
        print(resultA.shape)
        print(resultX.shape)
        pass

    def test_pad_compound_graph(self):
        smiles = get_smiles_string('AAAAA')
        A, X = get_compound_graph(smiles,  ['C', 'N', 'O', 'P'])
        resultA = pad_compound_graph([A], 133)
        resultX = pad_compound_graph([X], 133, axis=0)
        write_list_to_file("test_pad_compound_graph_resultA", resultA)
        write_list_to_file("test_pad_compound_graph_resultX", resultX)

        pass

    def test_get_AX_matrix(self):
        print("test_get_AX_matrix")
        pass

    def test_dataset_loader(self):
        print("test_dataset_loader")

if __name__ == '__main__':
    unittest.main()