import unittest
import sys
import script_find_maximum as sfm
import script_approximate_functions as saf

class TestIntegApproxFunc(unittest.TestCase):
    def test_script_approximate_functions(self):
        sys.argv = ['script_approximate_functions.py',
                    '--nEpoch', '500',
                    '--seed', '42',
                    '--model', 'picnn',
                    '--dataset', 'linear',
                    # '--noncvx', 'True',
                    '--noise', 'no_noise',
                    '--bounds', '0', '1', '0', '1',
                    '--num_data', '500']
        print("Running script_approximate_functions.py...")
        saf.main()
        print("Done.")
        # read the file and check the content
        with open('work/picnn.linear/train.csv', 'r') as f:
            # read the last line and get the validation loss
            last_line = f.readlines()[-2]
            val_loss = last_line.split(',')[1]
        val_loss = float(val_loss)
        print()
        print("Validation loss is ", val_loss)
        self.assertTrue(val_loss < 1e-2)
        print("Under the error tolerence in the order of O(10^(-3)), we accept the result.")
        print("*****************************************************************")

class TestIntegFindMax(unittest.TestCase):
    def test_script_find_maximum(self):
        sys.argv = ['script_find_maximum.py',
                    '--algo', 'gs',
                    '--d', '5',
                    '--L', '20',
                    '--lr', '0.01',
                    '--rounds', '100',
                    '--iters', '50',
                    '--c', '0.1',
                    '--seed', '42',
                    '--dataset', 'linear',
                    # '--noncvx', 'False',
                    '--num_data', '500',
                    '--bounds', '0', '1', '0', '1',
                    '--noise', 'no_noise']
        print("Running script_find_maximum.py...")
        sfm.main()
        print("Done.")
        # read the file and check the content
        with open('test/train.json', 'r') as f:
            # read the last line and get the loss
            last_line = f.readlines()[-1]
            last_loss = last_line.split(' ')[-1]

        last_loss = float(last_loss[:-2])
        print()
        print("Estimated maximum is ", last_loss)
        print("True maximum is ", 3.35161803)
        relative_err = abs((last_loss) - 3.35161803 )
        print("Error is ", relative_err)
        self.assertTrue(relative_err < 1e-1)
        print("Under the error tolerence in the order of O(10^(-2)), we accept the result.")
        print("*****************************************************************")

if __name__ == '__main__':
    unittest.main()