$ Unit system : kg, m, s  (SI)
$ Small steel coupon: MAT_PIECEWISE_LINEAR_PLASTICITY (MAT_024)
*KEYWORD
*TITLE
steel coupon - kg-m-s
*MAT_PIECEWISE_LINEAR_PLASTICITY
$     mid        ro         e        pr      sigy      etan      fail      tdel
         1    7850.02.10000E11       0.3  2.500E+8  1.000E+9       0.0       0.0
       0.0       0.0         0         0       0.0       0.0
*SECTION_SHELL
         1         2       1.0         2         1         0         0         1
     0.002     0.002     0.002     0.002       0.0       0.0       0.0         0
*PART
plate
         1         1         1         0         0         0         0         0
*NODE
       1             0.0             0.0             0.0       0       0
       2           0.010             0.0             0.0       0       0
       3           0.010           0.010             0.0       0       0
       4             0.0           0.010             0.0       0       0
*DEFINE_CURVE_TITLE
gravity g=9.80665 m/s^2
         1         0       1.0       1.0       0.0       0.0         0         0
                 0.0         9.80665
                 1.0         9.80665
*LOAD_BODY_Z
         1      -1.0         0       0.0       0.0       0.0         0
*END
