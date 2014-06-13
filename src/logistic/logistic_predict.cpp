#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include <string>
#include <sstream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>

#include "logistic.h"
#include "mlutils.h"

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include "Eigen/Core"
using namespace Eigen;

// global model variables
ProbSize m, n; // numbers of instances and features
ClassMap classmap; // a map of labels to label indices
LayerSize numlabels;
bool scaling = true;
std::string outputfile = "logistic.probas", orderfile = "logistic.order";

typedef std::vector<std::string> DataVec;


int main (int argc, char *argv[]) {
    // handle cmd args
	std::string datadir, model_file;

	if ( argc != 3 ) {
		printf( "Usage: ./clf_predict <data_directory> <model_file>\n" );
		exit( 0 );
	} else {
		datadir = argv[1];
		model_file = argv[2];
	}


	/* DATA PREPROCESSING */
	// determine number of instances
	DataVec datavec;
	mlu::count_instances( datadir, datavec, m );

	// determine number of features
	mlu::count_features( datavec[0], n );

	// record prediction order
	printf( "\nWriting Instance Order: %s\n", orderfile.c_str() );
	std::ofstream file( orderfile );
	if ( file1.is_open() ) {
		for ( auto& str : datavec ) {
			unsigned found = str.find_last_of("/");
  			file << str.substr(found+1) << "\t";
		}
		file << std::endl;
	}
	file.close();

	/* DATA INITIALIZATION */
    // danamically allocate data
	Mat X( m, n );
	Vec labels( m );

    // load testing data
    double feat_val, label;
	for ( ProbSize i=0; i<m; ++i ) {
        std::ifstream data( datavec[i] );
		for ( ProbSize j=0; j<n; ++j ) {
			data >> feat_val;
			X(i,j) = feat_val;
		}
		data >> label;
		labels[i] = label;
		//printf( "label %lf\n", label );
	}

    // perform feature scaling (optional)
    if ( scaling ) {
		mlu::scale_features( X, 1, 0 );
    }
	// std::cout << X << "\n";

	/* FORMAT LABELS */
	// format the local label set into a matrix based on global class map
	mlu::get_unique_labels( labels, classmap );
	Mat y = mlu::format_labels( labels, classmap );
	numlabels = (LayerSize) classmap.size();

	/* INIT LOCAL CLASSIFIER */
	LogisticRegression clf( n, numlabels );

	// load model
	Vec theta = clf.get_theta();
    std::ifstream model( model_file );
    int num_params;
    
    // validate paramater cardinality
    model >> num_params;
    if ( num_params != theta.size() ) {
    	printf( "ERROR: Model paramters cardinality does not match testing set.\n" );
    	return 1;
    }

    // load and set parameters
    for ( int i=0; i<num_params; ++i ) {
    	model >> theta[i];
    }
    clf.set_theta( theta );


    /* PREDICT */
   	// predict probabilities
   	printf( "Probability Matrix:\n" );
   	Mat probas = clf.predict_proba( X );
	std::cout << probas << std::endl;

	// predict class vector
	printf( "\nClass Prediction Vector:\n");
	Vec pred = clf.predict( X );
	std::cout << pred << std::endl;
	
	// output confusion matrix
	printf( "\nConfusion Matrix:\n" );
	Mat cm = mlu::confusion_matrix( y, pred );
	std::cout << cm << std::endl;


	/* OUTPUT PROBABILITIES */
	printf( "\nWriting Probabilities: %s\n", outputfile.c_str() );
	file( outputfile );
	if ( file.is_open() ) {
		file << probas << std::endl	;
	}
	file.close();


	return 0;
}
