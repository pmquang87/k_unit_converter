$ Unit system : ton, mm, s  (ton-mm-s, stress in MPa)
*KEYWORD
*TITLE
steel coupon - ton-mm-s
*MAT_PIECEWISE_LINEAR_PLASTICITY
$     mid        ro         e        pr      sigy      etan      fail      tdel
         1  7.850E-9 2.10000E5       0.3     250.0    1000.0       0.0       0.0
       0.0       0.0         0         0       0.0       0.0
*SECTION_SHELL
         1         2       1.0         2         1         0         0         1
       2.0       2.0       2.0       2.0       0.0       0.0       0.0         0
*PART
plate
         1         1         1         0         0         0         0         0
*NODE
       1             0.0             0.0             0.0       0       0
       2            10.0             0.0             0.0       0       0
       3            10.0            10.0             0.0       0       0
       4             0.0            10.0             0.0       0       0
*END
