#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>

#include "mpi.h"
#include "logistic.h"
#include "mlutils.h"

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include "Eigen/Core"
using namespace Eigen;

#define MASTER 0

typedef std::vector<std::string> DataVec;


// global model variables
ProbSize num_inst, m, n; // numbers of instance and features
size_t begin, end; // where to index the datavec for each partition
ClassMap classmap; // a map of labels to label indices
LayerSize numlabels;
double *delta_data;
double *X_min_ptr, *X_max_ptr, *X_min_data, *X_max_data;
bool scaling = true; // enable feature scaling


// MPI reduce ops
void reduce_unique_labels ( int *, int *, int *, MPI_Datatype * );
 
void reduce_unique_labels( int *invec, int *outvec, int *len, MPI_Datatype *dtype )
{
	int label;
    ClassSet merge;
    for ( int i=0; i<*len; ++i ) {
    	label = invec[i];
    	if ( label != -1 ) { merge.insert( label ); }
    }
    for ( auto& kv : classmap ) {
    	merge.insert( kv.first );
    }
    int idx = 0;
    for ( auto& elem : merge ) {
    	outvec[idx++] = elem;
    }
}


int main (int argc, char *argv[]) {
    // handle cmd args
	int batch_size, maxiter;
	std::string datadir;
	std::string output_file;

	if ( argc > 5 || argc < 2 ) {
		printf( "Usage: ./logistic_mpi <data_directory> <batch_size> "
				"<max_iterations> <model_output_file>\n");
		exit( 0 );
	} else if ( argc == 5 ) {
		datadir = argv[1];
		batch_size = atoi( argv[2] ); // mini-batch processing
		if ( batch_size == -1 ) { batch_size = INT_MIN; }
		maxiter = atoi( argv[3] );
		output_file = argv[4];
	} else if ( argc == 4 ) {
		datadir = argv[1];
		batch_size = atoi( argv[2] ); // mini-batch processing
		if ( batch_size == -1 ) { batch_size = INT_MIN; }
		maxiter = atoi( argv[3] );
		output_file = "logistic.model";
	} else if ( argc == 4 ) {
		datadir = argv[1];
		batch_size = atoi( argv[2] ); // mini-batch processing
		if ( batch_size == -1 ) { batch_size = INT_MIN; }
		maxiter = 100;
		output_file = "logistic.model";
	} else {
		datadir = argv[1];
		batch_size = INT_MIN; // batch processing
		maxiter = 100;
		output_file = "logistic.model";
	}

	// initialize/populate mpi specific vars local to each node
	double t1,t2, total; // elapsed time computation
	int  numtasks, taskid, len;
	char hostname[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	MPI_Get_processor_name(hostname, &len);
	MPI_Op op;


	/* DATA PREPROCESSING */
	if ( taskid == MASTER ) {
		printf( "\nLoading and Preprocessing Data\n" );
	}
	t1 = MPI_Wtime();

	// determine number of instances
	DataVec datavec;
	mlu::count_instances( datadir, datavec, num_inst );

	// determine number of features
	mlu::count_features( datavec[0], n );


	/* DATA INITIALIZATION */
	// randomize instances
	std::random_shuffle( datavec.begin(), datavec.end() );

	// partition data based on taskid
	size_t div = datavec.size() / numtasks;
	ProbSize limit = ( taskid == numtasks - 1 ) ? num_inst : div * ( taskid + 1 );
	m = limit - div * taskid;

    // danamically allocate data
	Mat X( m, n );
	Vec labels( m );

    // load data partition
    double feat_val, label;
    ProbSize i = 0;
	for ( ProbSize idx = taskid * div; idx < limit; ++idx ) {
	    std::ifstream data( datavec[idx] );
		for ( ProbSize j=0; j<n; ++j ) {
			data >> feat_val;
			X(i,j) = feat_val;
		}
		data >> label;
		labels[i] = label;
        i++;
	}

    // perform feature scaling (optional)
    if ( scaling ) {
    	// Allreduce to find global min
    	Vec X_min_tmp = X.colwise().minCoeff();
    	X_min_data = X_min_tmp.data();
    	Vec X_min = Vec( X_min_tmp.size() );
		MPI_Allreduce( X_min_tmp.data(), X_min.data(), X_min_tmp.size(), MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );

    	// Allreduce to find global max
		Vec X_max_tmp = X.colwise().maxCoeff();
		X_max_data = X_max_tmp.data();
		Vec X_max = Vec( X_max_tmp.size() );
		MPI_Allreduce( X_max_tmp.data(), X_max.data(), X_max_tmp.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

		// scale features using global min and max
		mlu::scale_features( X, X_min, X_max, 1, 0 );
    }


	/* FORMAT LABELS */
	// get unique labels
	mlu::get_unique_labels( labels, classmap );

	// allreduce to obtain maximum label set size
	int local_size = classmap.size();
	int max_size = 0;
	MPI_Allreduce( &local_size, &max_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

	// allreduce to obtain global unique label set
	int unique_labels[max_size];
	int global_unique_labels[max_size];
	for ( int i=0; i<max_size; ++i ) {
		unique_labels[i] = -1;
		global_unique_labels[i] = -1;
	}
	int idx = 0;
	for ( auto& kv : classmap ) {
		unique_labels[idx++] = kv.first;
	}
	MPI_Op_create( (MPI_User_function *)reduce_unique_labels, 1, &op );
	MPI_Allreduce( unique_labels, global_unique_labels, max_size, MPI_INT, op, MPI_COMM_WORLD );
	MPI_Op_free( &op );
	
	// update local classmap
	std::sort( global_unique_labels, global_unique_labels + max_size );
	classmap.clear();
	int labeltmp;
	idx=0;
	for ( int i=0; i<max_size; ++i ) {
		labeltmp = global_unique_labels[i];
		if ( labeltmp != -1 ) {
			classmap.emplace( labeltmp, idx++ );
		}
	}

	// format the local label set into a matrix based on global class map
	Mat y = mlu::format_labels( labels, classmap );
	numlabels = (LayerSize) classmap.size();

	// output total data loading time for each task
	MPI_Barrier( MPI_COMM_WORLD );
    t2 = MPI_Wtime();
	printf( "--- task %d loading time %lf\n", taskid, t2 - t1 ); 


	/* INIT LOCAL CLASSIFIER */
	LogisticRegression logistic_layer( n, numlabels, true );


	/* OPTIMIZATION */
	if ( taskid == MASTER ) {
		printf( "\nPerforming Gradient Descent\n" );
	}

	int update_size; // stores the number of instances read for each update
	double grad_mag; // stores the magnitude of the gradient for each update
	int delta_size = logistic_layer.get_theta_size();
	Vec delta_update = Vec::Zero( delta_size );
	int global_update_size;
	if ( taskid == MASTER ) {
		printf( "iteration : elapsed time : magnitude\n" );
	}

	total = MPI_Wtime(); // total training time marker
	for ( int i=0; i<maxiter; ++i ) {
		// compute gradient update
		t1 = MPI_Wtime();
		logistic_layer.compute_gradient( X, y, batch_size, update_size );
		delta_data = logistic_layer.get_delta().data();

		// sum updates across all partitions
		MPI_Allreduce( 
			delta_data,
			delta_update.data(),
			delta_size,
			MPI_DOUBLE,
			MPI_SUM,
			MPI_COMM_WORLD
		);
		logistic_layer.set_delta( delta_update );

		// sum the update sizes
		MPI_Allreduce( &update_size, &global_update_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

		// normalize + regularize gradient update
		logistic_layer.normalize_gradient( global_update_size );
		logistic_layer.regularize_gradient( global_update_size );

		// update logistic_layer parameters
		t2 = MPI_Wtime();
		if ( logistic_layer.converged( grad_mag ) ) { break; }
		if ( taskid == MASTER ) {
			printf( "%d : %lf : %lf\n", i+1, t2 - t1, grad_mag );
		}
		logistic_layer.update_theta();
	}
	t2 = MPI_Wtime();
	if ( taskid == MASTER ) {
		printf( "\nTotal Training Time %lf seconds\n", t2 - total );
	}


	/* MODEL STORAGE */
	if (taskid == MASTER) {
		FILE *output;
		output = fopen ( output_file.c_str(), "w" );
		int idx;
		Vec theta = logistic_layer.get_theta();
		printf( "\nWriting Model to File: %s\n\n", output_file.c_str() );

		fprintf( output, "%lu\n", theta.size() );
		for ( idx=0; idx<theta.size()-1; ++idx ) {
			fprintf( output, "%lf\t", theta[idx] );
		}
		fprintf( output, "%lf\n", theta[idx] );

		fclose( output );
	} 

	MPI_Finalize();
	return 0;
}
