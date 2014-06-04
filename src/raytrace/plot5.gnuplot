set term png
set output 'out5.png'

set yrange [-2:100]
set xrange [-15:15]

p_C = -0.128
p_k = -0.824
p_D0= 17.705
p_D2= 0.0211*0.0211
p_D4= 0.0871*0.0871*0.0871*0.0871

p_a = 10.0
p_ccd_z = 90.0
p_z(x) = ((p_C*x*x)/(1+sqrt(1-(1+p_k)*p_C*p_C*x*x))+p_D0+p_D2*x*x+p_D4*x*x*x*x)

z1(x) = p_a
z2(x) = p_z(x)
z3(x) = p_ccd_z - p_z(x)
z4(x) = p_ccd_z - p_a

plot z1(x), z2(x), z3(x), z4(x), 'testfinal.dat' u 1:3 w l