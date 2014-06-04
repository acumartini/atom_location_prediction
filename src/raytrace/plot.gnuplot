set term png
set output 'out.png'
set size square
z1(x) = (x*x)
z2(x) = 20.0
plot z1(x), 'testnorm.dat' u 1:3 w l

