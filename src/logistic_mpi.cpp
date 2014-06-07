#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include <string>
#include <dirent.h>
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
bool scaling = true;
double *X_min_ptr, *X_max_ptr, *X_min_data, *X_max_data;

// MPI reduce ops
void reduce_unique_labels ( int *, int *, int *, MPI_Datatype * );
void reduce_gradient_update ( double *, double *, int *, MPI_Datatype * );
void reduce_X_min ( double *, double *, int *, MPI_Datatype * );
void reduce_X_max ( double *, double *, int *, MPI_Datatype * );
 
void reduce_unique_labels( int *invec, int *inoutvec, int *len, MPI_Datatype *dtype )
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
    	inoutvec[idx++] = elem;
    }
}

void reduce_gradient_update( double *delta_in, double *delta_out, int *len, MPI_Datatype *dtype ) {
	for ( int i=0; i<*len; ++i ) {
        // printf( "len %d i %d delta_in[i] %f delta_data[i] %f\n", *len, i, delta_in[i], delta_data[i] );
		delta_out[i] = delta_in[i] + delta_data[i];
	}
}

void reduce_X_min( double *X_min_in, double *X_min_out, int *len, MPI_Datatype *dtype ) {
	for ( int i=0; i<*len; ++i ) {
		X_min_out[i] = std::min( X_min_in[i], X_min_data[i] );
	}
}

void reduce_X_max( double *X_max_in, double *X_max_out, int *len, MPI_Datatype *dtype ) {
	for ( int i=0; i<*len; ++i ) {
		X_max_out[i] = std::max( X_max_in[i], X_max_data[i] );
	}
}


void count_instances( std::string datadir, DataVec& datavec ) {
	struct dirent *pDirent;
	DIR *pDir;
	num_inst = 0;
	std::string filename;
	pDir = opendir( datadir.c_str() );

	if (pDir != NULL) {
	    while ( ( pDirent = readdir( pDir ) ) != NULL) {
	    	filename = pDirent->d_name;
	    	if ( filename != "." && filename != ".." ) {
	    		datavec.push_back( datadir + "/" + filename );
		    	num_inst++; 
	    	}
	   	}
	    closedir (pDir);
	}
}

void count_features( std::string datadir, int taskid ) {
	std::ifstream infile( datadir + "/0.tsv" );
	std::string line;
	std::getline( infile, line );
    std::istringstream iss( line );
    std::vector<std::string> tokens{
    	std::istream_iterator<std::string>{iss},
    	std::istream_iterator<std::string>{}
    };
    n = tokens.size() - 1; // -1 for label
}

int main (int argc, char *argv[]) {
    // handle cmd args
	int batch_size, maxiter;
	std::string datadir;
	std::string output_file;

	if ( argc > 5 || argc < 2 ) {
		printf( "Usage: ./logistic_mpi <data_directory> <batch_size> "
				"<max_iterations> <model_output_file>\n");
		MPI_Finalize();
		exit( 0 );
	} else if ( argc == 5 ) {
		datadir = argv[1];
		batch_size = atoi( argv[2] ); // mini-batch processing
		maxiter = atoi( argv[3] );
		output_file = argv[4];
	} else if ( argc == 4 ) {
		datadir = argv[1];
		batch_size = atoi( argv[2] ); // mini-batch processing
		maxiter = atoi( argv[3] );
		output_file = "clf.model";
	} else if ( argc == 4 ) {
		datadir = argv[1];
		batch_size = atoi( argv[2] ); // mini-batch processing
		maxiter = 100;
		output_file = "clf.model";
	} else {
		datadir = argv[1];
		batch_size = INT_MIN; // batch processing
		maxiter = 100;
		output_file = "clf.model";
	}

	// initialize/populate mpi specific vars local to each node
	int  numtasks, taskid, len, dest, source;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	MPI_Get_processor_name(hostname, &len);
	MPI_Op op;


	/* DATA PREPROCESSING */
	// determine number of instances
	DataVec datavec;
	count_instances( datadir, datavec );

	// determine number of features
	count_features( datadir, taskid );


	/* DATA INITIALIZATION */
	// randomize instance 
	std::random_shuffle ( datavec.begin(), datavec.end() );

	// partition data based on taskid
	size_t div = datavec.size() / numtasks;
	ProbSize limit = ( taskid == numtasks - 1 ) ? num_inst : div * ( taskid + 1 );
	m = limit - div * taskid;
	//printf( "m %lu n %lu\n", m, n );

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
            //printf( "taskid %d i %lu j %lu\n", taskid, i, j );
			X(i,j) = feat_val;
		}
		data >> label;
		labels[i] = label;
        i++;
	}
    // std::cout << X << "\n" << labels << "\n";

    // perform feature scaling (optional)
    if ( scaling ) {
    	// Allreduce to find global min
    	MPI_Op_create( (MPI_User_function *)reduce_X_min, 1, &op );
    	Vec X_min_tmp = X.colwise().minCoeff();
    	X_min_data = X_min_tmp.data();
    	Vec X_min = Vec( X_min_tmp.size() );
		MPI_Allreduce( X_min_tmp.data(), X_min.data(), X_min_tmp.size(), MPI_DOUBLE, op, MPI_COMM_WORLD );
		MPI_Op_free( &op );

    	// Allreduce to find global max
    	MPI_Op_create( (MPI_User_function *)reduce_X_max, 1, &op );
		Vec X_max_tmp = X.colwise().maxCoeff();
		X_max_data = X_max_tmp.data();
		Vec X_max = Vec( X_max_tmp.size() );
		MPI_Allreduce( X_max_tmp.data(), X_max.data(), X_max_tmp.size(), MPI_DOUBLE, op, MPI_COMM_WORLD );
		MPI_Op_free( &op );

		// std::cout << "\n" << X_min << "\n";
		// std::cout << X_max << "\n\n";

		// scale features using global min and max
		mlu::scale_features( X, X_min, X_max, 1, 0 );

		// std::cout << X << "\n" << y << "\n";
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
	for ( int i=0; i<max_size; ++i ) {
		unique_labels[i] = -1;
	}
	int idx = 0;
	for ( auto& kv : classmap ) {
		unique_labels[idx++] = kv.first;
	}
	int global_unique_labels[max_size];
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


	/* INIT LOCAL CLASSIFIER */
	LogisticRegression clf( n, numlabels, true );

	// initialize and communicate paramters
	if (taskid == MASTER) {
		// init and send parameters

	} else {
		// recieve network parameters and update local classifier

	}


	/* OPTIMIZATION */
	double grad_mag;
	int delta_size = clf.get_theta_size();
	Vec delta_update = Vec::Zero( delta_size );

	MPI_Op_create( (MPI_User_function *)reduce_gradient_update, 1, &op );
	for ( int i=0; i<maxiter; ++i ) {
		// compute gradient update
		clf.compute_gradient( X, y );
		delta_data = clf.get_delta().data();
        // std::cout << "TASK " << taskid << " DELTA " <<  clf.get_delta() << "\n\n";

		// sum updates across all partitions
		MPI_Allreduce( 
			delta_data,
			delta_update.data(),
			delta_size,
			MPI_DOUBLE,
			op,
			MPI_COMM_WORLD
		);
		clf.set_delta( delta_update );

		// normalize + regularize gradient update
		clf.normalize_gradient( num_inst );
		clf.regularize_gradient( num_inst );

		// update clf parameters
		if ( clf.converged( grad_mag ) ) { break; }
		if ( taskid == MASTER ) {
			printf( "%d : %lf\n", i+1, grad_mag );
		}
		clf.update_theta();
	}
	MPI_Op_free( &op );


	// perform prediction and model storage tasks on a single node
	if (taskid == MASTER) {
		/* MODEL STORAGE */
		FILE *output;
		output = fopen ( output_file, "w" );
		size_t idx;
		Vec theta = clf.get_theta();

		fprintf( output, "%lu\n", theta.size() );
		for ( idx=0; idx<theta_size-1; ++idx ) {
			fprintf( output, "%lf\t", theta[idx] );
		}
		fprintf( output, "%lf\n", theta[idx] );

		fclose( output );
	} 

	MPI_Finalize();
	return 0;
}
