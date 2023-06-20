import unittest
from Molecule import *
from numpy.linalg import LinAlgError

class MoleculeTest(unittest.TestCase):

    def test_3D_default(self):
        smiles = """
     RDKit          3D

136137  0  0  0  0  0  0  0  0999 V2000
    6.6924   -0.5616   -3.8514 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0599    0.7911   -3.5232 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0703    1.6800   -4.7698 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.7927    1.5045   -2.3697 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.6785    0.8260   -0.9893 C   0  0  2  0  0  0  0  0  0  0  0  0
    5.3697    1.0567   -0.3870 N   0  0  0  0  0  0  0  0  0  0  0  0
    4.3116    0.2002   -0.5218 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.3530   -0.8283   -1.2031 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.0553    0.5983    0.2655 C   0  0  2  0  0  0  0  0  0  0  0  0
    3.1879    0.0929    1.6989 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.4010    0.9344    2.6534 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5351    0.5320    3.4134 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.7756    2.2276    2.6557 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.9063    0.0112   -0.4252 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.6483    0.5606   -0.3653 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4021    1.5980    0.2467 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4233   -0.1969   -1.1629 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.2393    0.0952   -2.6646 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2010   -0.6329   -3.5784 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2937   -2.0311   -3.5618 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1835   -2.6944   -4.4105 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.9727   -1.9719   -5.2990 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8775   -0.5825   -5.3380 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9958    0.0842   -4.4837 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7325    0.2433   -0.6776 N   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7172   -0.6608   -0.3642 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5761   -1.8781   -0.5148 O   0  0  0  0  0  0  0  0  0  0  0  0
   -4.0611   -0.0725    0.1057 C   0  0  2  0  0  0  0  0  0  0  0  0
   -3.9616    0.6621    1.4448 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.5391   -0.2057    2.4919 O   0  0  0  0  0  0  0  0  0  0  0  0
   -4.9897   -1.2031    0.1661 N   0  0  0  0  0  0  0  0  0  0  0  0
   -6.3485   -1.0600    0.0286 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.9155    0.0336    0.0292 O   0  0  0  0  0  0  0  0  0  0  0  0
   -7.0772   -2.3970   -0.2212 C   0  0  2  0  0  0  0  0  0  0  0  0
   -6.9239   -2.7739   -1.6994 C   0  0  0  0  0  0  0  0  0  0  0  0
   -8.2908   -3.3308   -2.0761 C   0  0  0  0  0  0  0  0  0  0  0  0
   -9.2529   -2.5001   -1.2407 C   0  0  0  0  0  0  0  0  0  0  0  0
   -8.5151   -2.2338   -0.0198 N   0  0  0  0  0  0  0  0  0  0  0  0
   -8.9885   -1.9822    1.2525 C   0  0  0  0  0  0  0  0  0  0  0  0
   -8.2602   -2.1609    2.2345 O   0  0  0  0  0  0  0  0  0  0  0  0
  -10.3512   -1.2720    1.4181 C   0  0  2  0  0  0  0  0  0  0  0  0
  -11.5688   -2.0129    0.8634 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.9087   -1.5831    1.4640 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.9676   -1.8503    2.9485 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.0320   -0.7345    3.7298 N   0  0  0  0  0  0  0  0  0  0  0  0
  -12.9324   -2.9766    3.4318 O   0  0  0  0  0  0  0  0  0  0  0  0
  -10.1751    0.0542    0.8196 N   0  0  0  0  0  0  0  0  0  0  0  0
  -10.6800    1.1943    1.4015 C   0  0  0  0  0  0  0  0  0  0  0  0
  -11.4815    1.1647    2.3368 O   0  0  0  0  0  0  0  0  0  0  0  0
  -10.2277    2.5325    0.7729 C   0  0  1  0  0  0  0  0  0  0  0  0
   -9.1247    2.3863   -0.2043 N   0  0  0  0  0  0  0  0  0  0  0  0
  -11.4528    3.2492    0.1724 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.1047    2.4600   -1.3461 S   0  0  0  0  0  0  0  0  0  0  0  0
    7.7238    1.4371   -0.0294 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.3951    2.2440    0.8375 O   0  0  0  0  0  0  0  0  0  0  0  0
    9.0282    1.0494   -0.2851 N   0  0  0  0  0  0  0  0  0  0  0  0
   10.1369    1.2930    0.6549 C   0  0  1  0  0  0  0  0  0  0  0  0
   10.1510    2.7123    1.2484 C   0  0  0  0  0  0  0  0  0  0  0  0
   11.5491    3.2304    1.5979 C   0  0  0  0  0  0  0  0  0  0  0  0
   12.0242    2.7762    2.9582 C   0  0  0  0  0  0  0  0  0  0  0  0
   13.3831    2.7587    3.0941 N   0  0  0  0  0  0  0  0  0  0  0  0
   11.2957    2.4169    3.8720 O   0  0  0  0  0  0  0  0  0  0  0  0
   10.1245    0.1498    1.7017 C   0  0  0  0  0  0  0  0  0  0  0  0
    9.0925   -0.4344    2.0319 O   0  0  0  0  0  0  0  0  0  0  0  0
   11.3668   -0.2206    2.1771 N   0  0  0  0  0  0  0  0  0  0  0  0
   11.5037   -1.2330    3.2321 C   0  0  1  0  0  0  0  0  0  0  0  0
   12.5036   -2.3353    2.8571 C   0  0  0  0  0  0  0  0  0  0  0  0
   12.0106   -3.2807    1.3740 S   0  0  0  0  0  0  0  0  0  0  0  0
   12.0525   -0.5469    4.4810 C   0  0  0  0  0  0  0  0  0  0  0  0
   13.1217    0.0431    4.5525 O   0  0  0  0  0  0  0  0  0  0  0  0
   11.2405   -0.6563    5.5507 O   0  0  0  0  0  0  0  0  0  0  0  0
    6.6273   -1.2552   -3.0102 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.1813   -1.0346   -4.6959 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.7485   -0.4505   -4.1180 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.0125    0.6266   -3.2537 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.0886    1.9526   -5.0586 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.5990    1.1717   -5.6168 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.5101    2.6020   -4.5874 H   0  0  0  0  0  0  0  0  0  0  0  0
    7.8510    1.6043   -2.6467 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.4107    2.5311   -2.2863 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.8582   -0.2512   -1.0444 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.3340    1.8089    0.3013 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.9559    1.6895    0.2115 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.8394   -0.9455    1.7807 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.2292    0.1036    2.0456 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1569    2.6413    3.2962 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.0782   -0.8481   -0.9415 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2944   -1.2646   -0.9535 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3283    1.1786   -2.8291 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7808   -0.1684   -2.9800 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6897   -2.6196   -2.8764 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2539   -3.7771   -4.3727 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.6592   -2.4896   -5.9620 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4863   -0.0174   -6.0379 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9342    1.1676   -4.5361 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7662    1.1809   -0.2902 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.4156    0.6174   -0.6722 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.9311    1.0771    1.7349 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.2502    1.4914    1.3898 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.1782   -0.9378    2.5267 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.5611   -2.1111   -0.0038 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.7030   -3.1774    0.4515 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.1194   -3.4965   -1.8578 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.7024   -1.8938   -2.3177 H   0  0  0  0  0  0  0  0  0  0  0  0
   -8.3479   -4.3852   -1.7777 H   0  0  0  0  0  0  0  0  0  0  0  0
   -8.4953   -3.2632   -3.1485 H   0  0  0  0  0  0  0  0  0  0  0  0
  -10.1666   -3.0617   -1.0492 H   0  0  0  0  0  0  0  0  0  0  0  0
   -9.4829   -1.5419   -1.7177 H   0  0  0  0  0  0  0  0  0  0  0  0
  -10.4687   -1.1390    2.5000 H   0  0  0  0  0  0  0  0  0  0  0  0
  -11.4371   -3.0899    1.0203 H   0  0  0  0  0  0  0  0  0  0  0  0
  -11.6510   -1.8494   -0.2160 H   0  0  0  0  0  0  0  0  0  0  0  0
  -13.1145   -0.5277    1.2489 H   0  0  0  0  0  0  0  0  0  0  0  0
  -13.7205   -2.1560    0.9962 H   0  0  0  0  0  0  0  0  0  0  0  0
  -12.9229   -0.9027    4.7244 H   0  0  0  0  0  0  0  0  0  0  0  0
  -12.5754    0.1085    3.3777 H   0  0  0  0  0  0  0  0  0  0  0  0
   -9.3509    0.1766    0.2399 H   0  0  0  0  0  0  0  0  0  0  0  0
   -9.8443    3.1386    1.6048 H   0  0  0  0  0  0  0  0  0  0  0  0
   -9.4641    1.8756   -1.0186 H   0  0  0  0  0  0  0  0  0  0  0  0
   -8.3369    1.8693    0.1953 H   0  0  0  0  0  0  0  0  0  0  0  0
  -11.1867    4.2822   -0.0819 H   0  0  0  0  0  0  0  0  0  0  0  0
  -12.2715    3.2869    0.8991 H   0  0  0  0  0  0  0  0  0  0  0  0
  -13.1334    3.3051   -1.5168 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.1190    0.1931   -0.8200 H   0  0  0  0  0  0  0  0  0  0  0  0
   11.0319    1.1495    0.0357 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.4994    2.7649    2.1244 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.7476    3.4060    0.4990 H   0  0  0  0  0  0  0  0  0  0  0  0
   11.5289    4.3283    1.6250 H   0  0  0  0  0  0  0  0  0  0  0  0
   12.2753    2.9347    0.8306 H   0  0  0  0  0  0  0  0  0  0  0  0
   13.7103    2.4988    4.0224 H   0  0  0  0  0  0  0  0  0  0  0  0
   13.9541    3.4176    2.5838 H   0  0  0  0  0  0  0  0  0  0  0  0
   12.1727    0.3689    2.0079 H   0  0  0  0  0  0  0  0  0  0  0  0
   10.5272   -1.6771    3.4663 H   0  0  0  0  0  0  0  0  0  0  0  0
   13.4935   -1.9104    2.6548 H   0  0  0  0  0  0  0  0  0  0  0  0
   12.6168   -3.0420    3.6870 H   0  0  0  0  0  0  0  0  0  0  0  0
   13.1026   -4.0574    1.3211 H   0  0  0  0  0  0  0  0  0  0  0  0
   11.6040   -0.0257    6.2092 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  2  4  1  0
  4  5  1  0
  5  6  1  0
  6  7  1  0
  7  8  2  0
  7  9  1  0
  9 10  1  0
 10 11  1  0
 11 12  2  0
 11 13  1  0
  9 14  1  0
 14 15  1  0
 15 16  2  0
 15 17  1  0
 17 18  1  0
 18 19  1  0
 19 20  2  0
 20 21  1  0
 21 22  2  0
 22 23  1  0
 23 24  2  0
 17 25  1  0
 25 26  1  0
 26 27  2  0
 26 28  1  0
 28 29  1  0
 29 30  1  0
 28 31  1  0
 31 32  1  0
 32 33  2  0
 32 34  1  0
 34 35  1  0
 35 36  1  0
 36 37  1  0
 37 38  1  0
 38 39  1  0
 39 40  2  0
 39 41  1  0
 41 42  1  0
 42 43  1  0
 43 44  1  0
 44 45  1  0
 44 46  2  0
 41 47  1  0
 47 48  1  0
 48 49  2  0
 48 50  1  0
 50 51  1  0
 50 52  1  0
 52 53  1  0
  5 54  1  0
 54 55  2  0
 54 56  1  0
 56 57  1  0
 57 58  1  0
 58 59  1  0
 59 60  1  0
 60 61  1  0
 60 62  2  0
 57 63  1  0
 63 64  2  0
 63 65  1  0
 65 66  1  0
 66 67  1  0
 67 68  1  0
 66 69  1  0
 69 70  2  0
 69 71  1  0
 24 19  1  0
 38 34  1  0
  1 72  1  0
  1 73  1  0
  1 74  1  0
  2 75  1  0
  3 76  1  0
  3 77  1  0
  3 78  1  0
  4 79  1  0
  4 80  1  0
  5 81  1  6
  6 82  1  0
  9 83  1  1
 10 84  1  0
 10 85  1  0
 13 86  1  0
 14 87  1  0
 17 88  1  6
 18 89  1  0
 18 90  1  0
 20 91  1  0
 21 92  1  0
 22 93  1  0
 23 94  1  0
 24 95  1  0
 25 96  1  0
 28 97  1  6
 29 98  1  0
 29 99  1  0
 30100  1  0
 31101  1  0
 34102  1  1
 35103  1  0
 35104  1  0
 36105  1  0
 36106  1  0
 37107  1  0
 37108  1  0
 41109  1  1
 42110  1  0
 42111  1  0
 43112  1  0
 43113  1  0
 45114  1  0
 45115  1  0
 47116  1  0
 50117  1  1
 51118  1  0
 51119  1  0
 52120  1  0
 52121  1  0
 53122  1  0
 56123  1  0
 57124  1  6
 58125  1  0
 58126  1  0
 59127  1  0
 59128  1  0
 61129  1  0
 61130  1  0
 65131  1  0
 66132  1  1
 67133  1  0
 67134  1  0
 68135  1  0
 71136  1  0
M  END
    """
        expandAtom = True
        includeHydros = False
        fragment = True
        addRingNeighbors = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles, fragment, addRingNeighbors)
        output = find_ellipses(programInput)
        self.assertEqual(6, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(2.1086, magnitudes[0], 3)
        self.assertAlmostEqual(3.4459, magnitudes[1], 3)
        self.assertAlmostEqual(4.0556, magnitudes[2], 3)

    def test_comparison_ethane(self):
        smiles = """
     RDKit          3D

  8  7  0  0  0  0  0  0  0  0999 V2000
   -0.7558    0.0071   -0.0160 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7558   -0.0071    0.0160 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1627   -0.1018    0.9937 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1225    0.9487   -0.4356 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1350   -0.8147   -0.6307 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.1350    0.8148    0.6307 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.1627    0.1018   -0.9937 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.1226   -0.9487    0.4356 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
  2  6  1  0
  2  7  1  0
  2  8  1  0
M  END"""
        expandAtom = True
        includeHydros = False
        fragment = True
        addRingNeighbors = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles, fragment, addRingNeighbors)
        output = find_ellipses(programInput)
        self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.7854889, magnitudes[0], 6)
        self.assertAlmostEqual(1.7863160, magnitudes[1], 6)
        self.assertAlmostEqual(2.457547, magnitudes[2], 6)
    
    def test_ethane(self):
        smiles = 'CC'
        expandAtom = True
        includeHydros = False
        fragment = True
        addRingNeighbors = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles, fragment, addRingNeighbors)
        output = find_ellipses(programInput)
        self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.785, magnitudes[0], 1)
        self.assertAlmostEqual(1.786, magnitudes[1], 1)
        self.assertAlmostEqual(2.457, magnitudes[2], 0)


    def test_3D_benzene(self):
        smiles = """
     RDKit          3D

 12 12  0  0  0  0  0  0  0  0999 V2000
    0.8038   -1.1399    0.0149 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3891    0.1261   -0.0021 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5854    1.2659   -0.0170 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8038    1.1399   -0.0149 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3891   -0.1261    0.0021 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5854   -1.2659    0.0170 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4300   -2.0279    0.0265 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.4714    0.2243   -0.0038 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0414    2.2523   -0.0302 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4300    2.0279   -0.0265 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4714   -0.2243    0.0038 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0414   -2.2523    0.0302 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
  1  7  1  0
  2  8  1  0
  3  9  1  0
  4 10  1  0
  5 11  1  0
  6 12  1  0
M  END"""
        expandAtom = True
        includeHydros = False
        fragment = True
        addRingNeighbors = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles, fragment, addRingNeighbors)
        output = find_ellipses(programInput)
        #self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.9302225, magnitudes[0], 6)
        self.assertAlmostEqual(2.9402652, magnitudes[1], 6)
        self.assertAlmostEqual(3.1391175, magnitudes[2], 6)
        number_atoms = output.mol.GetNumAtoms();
        
    def test_benzene(self):
        smiles = "c1ccccc1"
        expandAtom = False
        includeHydros = False
        fragment = True
        addRingNeighbors = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles, fragment, addRingNeighbors)
        output = find_ellipses(programInput)
        self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.257e-05, magnitudes[0], 2)
        self.assertAlmostEqual(1.482, magnitudes[1], 1)
        self.assertAlmostEqual(1.899, magnitudes[2], 1)
        number_atoms = output.mol.GetNumAtoms();
        self.assertEqual(number_atoms - 6, len(ellipse.points))
    
    def test_cyanide_3D(self):
        smiles = """
     RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
   -0.0317    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.1283   -0.0002    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0967    0.0002    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  3  0
  1  3  1  0
M  END
"""
        expandAtom = True
        includeHydros = False
        fragment = True
        addRingNeighbors = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles, fragment, addRingNeighbors)
        output = find_ellipses(programInput)
        self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.7499138, magnitudes[0], 6)
        self.assertAlmostEqual(1.7501013, magnitudes[1], 6)
        self.assertAlmostEqual(2.2294569, magnitudes[2], 6)
    
    def test_cyanide_2D(self):
        smiles = "C#N"
        expandAtom = True
        includeHydros = False
        fragment = True
        addRingNeighbors = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles, fragment, addRingNeighbors)
        output = find_ellipses(programInput)
        self.assertEqual(1, len(output.ellipsis))
        ellipse = output.ellipsis[0]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.7499, magnitudes[0], 1)
        self.assertAlmostEqual(1.7501, magnitudes[1], 1)
        self.assertAlmostEqual(2.2294, magnitudes[2], 1)


    def test_ethane_without_hydrogens(self):
        smiles = 'CC'
        expandAtom = False
        includeHydros = False
        fragment = True
        addRingNeighbors = True 
        programInput = ProgramInput(includeHydros, expandAtom, smiles, fragment, addRingNeighbors)
        with self.assertRaises(ValueError):
            output = quadratic_to_parametric(programInput, NDArray)
    
    def test_smiles1(self):
        smiles = 'Cc1c(cc([nH]1)C(=O)NC2CCN(CC2)c3ccc4ccccc4n3)Br'
        expandAtom = True
        includeHydros = False
        fragment = True
        addRingNeighbors = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles, addRingNeighbors, fragment,)
        output = find_ellipses(programInput)
        ellipsis = output.ellipsis
        self.assertEqual(len(ellipsis), 4)

    def test_smiles1_3D(self):
        smiles = """
     RDKit          3D

 47 50  0  0  0  0  0  0  0  0999 V2000
   -6.9727   -1.7539    0.6903 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.3847   -0.6957   -0.1675 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.9700    0.2283   -0.9999 C   0  0  0  0  0  0  0  0  0  0  0  0
   -5.9606    1.0212   -1.5745 C   0  0  0  0  0  0  0  0  0  0  0  0
   -4.7421    0.5565   -1.1027 C   0  0  0  0  0  0  0  0  0  0  0  0
   -5.0288   -0.4718   -0.2270 N   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4229    1.0676   -1.4102 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.2291    2.1855   -1.8749 O   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3931    0.1784   -1.1620 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0092    0.5033   -1.5160 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4113    1.5728   -0.5883 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.3016    0.9664    0.6173 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3262   -0.0011    0.2076 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.2903   -0.4893   -1.1722 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1578   -0.7734   -1.5536 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.6003    0.1100    0.7688 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.7605    0.3844    2.1256 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.0381    0.4754    2.6732 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.1515    0.2805    1.8565 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.4644    0.3625    2.3545 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.5508    0.1568    1.5071 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.3286   -0.1313    0.1655 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0201   -0.2119   -0.3217 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.9189   -0.0103    0.5003 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.6801   -0.0985   -0.0181 N   0  0  0  0  0  0  0  0  0  0  0  0
   -8.7976    0.4595   -1.3461 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -7.6055   -1.3089    1.4650 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.1976   -2.3472    1.1867 H   0  0  0  0  0  0  0  0  0  0  0  0
   -7.5856   -2.4356    0.0916 H   0  0  0  0  0  0  0  0  0  0  0  0
   -6.0832    1.8452   -2.2672 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.3376   -0.9639    0.3232 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.6433   -0.7994   -1.1346 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0434    0.9275   -2.5284 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1706    2.2830   -0.2431 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.3201    2.1595   -1.1606 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4244    0.4548    1.2604 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7229    1.7953    1.1991 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8529   -1.4273   -1.2625 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.7316    0.2397   -1.8639 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5621   -1.5270   -0.8639 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2060   -1.2170   -2.5553 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8990    0.5066    2.7725 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.1514    0.6897    3.7319 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.6440    0.5866    3.4033 H   0  0  0  0  0  0  0  0  0  0  0  0
    8.5650    0.2208    1.8924 H   0  0  0  0  0  0  0  0  0  0  0  0
    8.1673   -0.2941   -0.5063 H   0  0  0  0  0  0  0  0  0  0  0  0
    5.8535   -0.4374   -1.3722 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  3  4  1  0
  4  5  2  0
  5  6  1  0
  5  7  1  0
  7  8  2  0
  7  9  1  0
  9 10  1  0
 10 11  1  0
 11 12  1  0
 12 13  1  0
 13 14  1  0
 14 15  1  0
 13 16  1  0
 16 17  2  0
 17 18  1  0
 18 19  2  0
 19 20  1  0
 20 21  2  0
 21 22  1  0
 22 23  2  0
 23 24  1  0
 24 25  2  0
  3 26  1  0
  6  2  1  0
 15 10  1  0
 25 16  1  0
 24 19  1  0
  1 27  1  0
  1 28  1  0
  1 29  1  0
  4 30  1  0
  6 31  1  0
  9 32  1  0
 10 33  1  0
 11 34  1  0
 11 35  1  0
 12 36  1  0
 12 37  1  0
 14 38  1  0
 14 39  1  0
 15 40  1  0
 15 41  1  0
 17 42  1  0
 18 43  1  0
 20 44  1  0
 21 45  1  0
 22 46  1  0
 23 47  1  0
M  END
"""
        expandAtom = True
        includeHydros = False
        fragment = True
        addRingNeighbors = True
        programInput = ProgramInput(includeHydros, expandAtom, smiles, addRingNeighbors, fragment,)
        output = find_ellipses(programInput)
        ellipsis = output.ellipsis
        ellipse = output.ellipsis[0]
        self.assertEqual(len(ellipsis), 4)
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.9608668, magnitudes[0], 6)
        self.assertAlmostEqual(4.3398075, magnitudes[1], 6)
        self.assertAlmostEqual(4.7940147, magnitudes[2], 6)
        ellipse = output.ellipsis[1]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(2.3278352, magnitudes[0], 6)
        self.assertAlmostEqual(3.38101004, magnitudes[1], 6)
        self.assertAlmostEqual(3.7063896, magnitudes[2], 6)
        ellipse = output.ellipsis[2]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(2.0768646, magnitudes[0], 6)
        self.assertAlmostEqual(3.2698849, magnitudes[1], 6)
        self.assertAlmostEqual(4.3541148, magnitudes[2], 6)
        ellipse = output.ellipsis[3]
        magnitudes = ellipse.axes_magnitudes
        self.assertAlmostEqual(1.5499999, magnitudes[0], 6)
        self.assertAlmostEqual(1.55, magnitudes[1], 6)
        self.assertAlmostEqual(1.5500000, magnitudes[2], 6)



    def test_smiles2(self):
        smiles = 'Fc1ccc(cc1)[C@@]3(OCc2cc(C#N)ccc23)CCCN(C)C'
        expandAtom = True
        includeHydros = False
        fragment = True
        addRingNeighbors = True 
        programInput = ProgramInput(includeHydros, expandAtom, smiles, addRingNeighbors, fragment,)
        output = find_ellipses(programInput)
        ellipsis = output.ellipsis
        self.assertEqual(len(ellipsis), 4)

    def test_smiles3(self):
        smiles = 'CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CO)NC(=O)[C@@H]1CCCN1C(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](N)CS)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CS)C(=O)O'
        expandAtom = True
        includeHydros = False
        fragment = True
        addRingNeighbors = True 
        programInput = ProgramInput(includeHydros, expandAtom, smiles, addRingNeighbors, fragment,)
        output = find_ellipses(programInput)
        ellipsis = output.ellipsis
        self.assertEqual(len(ellipsis), 6)




if __name__ == "__main__":
    unittest.main()

