#!/bin/bash
icc -o a_cilk main.cpp -lcilkrts
./a_cilk > testfinal.dat
gnuplot plot5.gnuplot
