#include <iostream>

#include "../logistic.h"
#include "../csv.h"
#include "../mlutils.h"

using namespace std;

int main () {
	// init multiclass LR 
	LogisticRegression clf( 13, 3 );

	// load csv data
	Mat X( 178, 13 );
	Vec y( 178 );
	io::CSVReader<14> in("../data/wine.csv");
	//in.read_header(io::ignore_extra_column, "vendor", "size", "speed");
	double a,b,c,d,e,f,g,h,i,j,k,l,m,clss;
	int row = 0;
	while( in.read_row( a,b,c,d,e,f,g,h,i,j,k,l,m,clss ) ){
		X( row, 0 ) = a;
		X( row, 1 ) = b;
		X( row, 2 ) = c;
		X( row, 3 ) = d;
		X( row, 4 ) = e;
		X( row, 5 ) = f;
		X( row, 6 ) = g;
		X( row, 7 ) = h;
		X( row, 8 ) = i;
		X( row, 9 ) = j;
		X( row, 10 ) = k;
		X( row, 11 ) = l;
		X( row, 12 ) = m;
		y[row] = clss;
		row++;
	}

	cout << X << endl << endl;
	cout << y << endl << endl;
	
	//mlu::scale_features( X, 1, 0 );
	//cout << X << endl << endl;

	// format labels
	Mat labels = mlu::format_labels( y );
	cout << labels << endl << endl;

	// fit dataset
	int maxiter = 10;

}