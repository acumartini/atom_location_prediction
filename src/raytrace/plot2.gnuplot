set term png
set output 'out2.png'
set size square
z1(x) = (x*x)

plot z1(x), 'testsnell.dat' u 1:3 w l