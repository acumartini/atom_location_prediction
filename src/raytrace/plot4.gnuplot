set term png
set output 'out4.png'

set yrange [-2:200]
set xrange [-5:5]
z1(x) = (x*x)/10+1
z2(x) = 10
z3(x) = -(x*x)/5.0+13.0

plot z1(x), z2(x), 'testrefract.dat' u 1:3 w l