#!/bin/bash
g++ -o a2 test_pointer.cpp
g++ -o a3 test_snell.cpp
g++ -o a4 test_intersect.cpp
icpc -o a5 test_refract.cpp -fopenmp
g++ -o a6 main.cpp
./a2 > testnorm.dat
./a3 > testsnell.dat
./a4 > testintersect.dat
./a5 > testrefract.dat
./a6 > testfinal.dat
gnuplot plot.gnuplot
gnuplot plot2.gnuplot
gnuplot plot3.gnuplot
gnuplot plot4.gnuplot
gnuplot plot5.gnuplot
