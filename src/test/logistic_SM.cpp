#include <iostream>
#include <assert.h>

#include "../logistic.h"
#include "../csv.h"
#include "../mlutils.h"

using namespace std;

int main () {
	// init multiclass LR 
	LogisticRegression clf( 13, 3 );

	// load csv data
	Mat X( 89, 13 );
	Vec labels( 89 );
	io::CSVReader<14> in("../data/train.csv");
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
		labels[row] = clss;
		row++;
	}

	mlu::scale_features( X, 1, 0 );
	cout << X << endl << endl;
	cout << labels << endl << endl;
	
	// format labels
	ClassMap classmap;
	mlu::get_unique_labels( labels, classmap );
	Mat y = mlu::format_labels( labels, classmap );
	// cout << labels << endl;
	// cout << labels.rows() << endl;

	// fit dataset
	int maxiter = 100000;

	// clf.compute_gradient_update( X, labels );
	double grad_mag;
	printf( "Interation : Magnitude\n" );
	for ( int i = 0; i < maxiter; ++i ) {
		clf.compute_gradient( X, y );
		if ( clf.converged( grad_mag ) ) { break; }
		printf( "%d : %lf\n", i, grad_mag );
		clf.update_theta();
	}

	Mat probas = clf.predict_proba( X );
	cout << probas << endl;
	Vec pred = clf.predict( X );
	cout << pred << endl;
	
	Mat cm = mlu::confusion_matrix( y, pred );
	cout << cm << endl;


	// try test data
	Mat X_test( 45, 13 );
	Vec labels_( 45 );
	io::CSVReader<14> in_test("../data/test.csv");
	//in_test.read_header(io::ignore_eX_testtra_column, "vendor", "size", "speed");
	//double a,b,c,d,e,f,g,h,i,j,k,l,m,clss;
	row = 0;
	while( in_test.read_row( a,b,c,d,e,f,g,h,i,j,k,l,m,clss ) ){
		X_test( row, 0 ) = a;
		X_test( row, 1 ) = b;
		X_test( row, 2 ) = c;
		X_test( row, 3 ) = d;
		X_test( row, 4 ) = e;
		X_test( row, 5 ) = f;
		X_test( row, 6 ) = g;
		X_test( row, 7 ) = h;
		X_test( row, 8 ) = i;
		X_test( row, 9 ) = j;
		X_test( row, 10 ) = k;
		X_test( row, 11 ) = l;
		X_test( row, 12 ) = m;
		labels_[row] = clss;
		row++;
	}

	mlu::scale_features( X_test, 1, 0 );
	cout << X_test << endl << endl;
	cout << labels_ << endl << endl;	

	// format labels
	Mat y_test = mlu::format_labels( labels_, classmap );
	cout << y_test << endl << endl;

	probas = clf.predict_proba( X_test );
	cout << probas << endl;
	pred = clf.predict( X_test );
	cout << pred << endl;
	
	cm = mlu::confusion_matrix( y_test, pred );
	cout << cm << endl;
}