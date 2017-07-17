import unittest
from loudness_lufs import r128Stats
from loudness_lufs import linearGain

"""
The following script serves to test the two helper functions in loudness_lufs.py
"""

class loudnessLufsTestCase(unittest.TestCase):
    """Test for 'loudness_lufs.py'"""

    def test_r128Stats(self):
        """Are the following three tests matched with the answers"""
        a = {'I': -37.0, 'gain': 11.220184543019636, 'LRA High': -37.7, 'LRA Threshold': -59.0, 'LRA Low': -41.3, 'LRA': 3.5, 'I Threshold': -47.1}
        b = {'LRA': 3.4, 'gain': 2.4547089156850306, 'LRA Threshold': -44.3, 'LRA High': -23.4, 'LRA Low': -26.8, 'I Threshold': -34.6, 'I': -23.8}
        c = {'I': -23.2, 'I Threshold': -33.9, 'gain': 2.290867652767773, 'LRA Threshold': -43.6, 'LRA Low': -26.0, 'LRA High': -22.7, 'LRA': 3.4}

        self.assertEqual(r128Stats('/home/ziang/trident/non_silence14.wav', stream=0)['I'], a['I'])
        self.assertEqual(r128Stats('/home/ziang/trident/man1_nb.wav', stream=0)['I'], b['I'])
        self.assertEqual(r128Stats('/home/ziang/trident/man1_wb.wav', stream=0)['I'], c['I'])

#        self.assertEqual(r128Stats('/home/ziang/trident/non_silence14.wav', stream=0), -37.0)
#        self.assertEqual(r128Stats('/home/ziang/trident/man1_nb.wav', stream=0), -23.8)
#        self.assertEqual(r128Stats('/home/ziang/trident/man1_wb.wav', stream=0), -23.2)

    def test_linearGain(self):
        # Test with previous three test cases
        self.assertEqual(round(linearGain(-37.0, -16), 12), round(11.220184543019636, 12))
        self.assertEqual(round(linearGain(-23.8, -16), 12), round(2.4547089156850306, 12))
        self.assertEqual(round(linearGain(-23.2, -16), 12), round(2.290867652767773, 12))

        # Test with values
        self.assertEqual(round(linearGain(-20.0, -16), 12), round(1.5848931924611135, 12))
        self.assertEqual(round(linearGain(-25.0, -16), 12), round(2.8183829312644538, 12))
        self.assertEqual(round(linearGain(-30.0, -16), 12), round(5.0118723362727228, 12))


if __name__ == '__main__':
    unittest.main() 