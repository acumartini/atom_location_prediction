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

#define  MASTER		0

typedef unsigned long ProbSize;

// globla model variables
ProbSize m, n; // numbers of instance and features
ClassMap classmap; // a map of labels to label indices
LayerSize numlabels;

// MPI reduce ops
void reduce_unique_labels ( int *, int *, int *, MPI_Datatype * );
// void reduce_gradient_update ( int *, int *, int *, MPI_Datatype * );
 
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


void count_instances( std::string datafile ) {
	struct dirent *pDirent;
	DIR *pDir;
	m = 0;
	pDir = opendir( datafile.c_str() );

	if (pDir != NULL) {
	    while ( ( pDirent = readdir( pDir ) ) != NULL) { m++; }
		m -= 2; // ignore "." and ".." directory reads
	    closedir (pDir);
	}
}

void count_features( std::string datafile, int taskid ) {
	std::ifstream infile( datafile );
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
	int batch_size;
	if ( argc > 2 ) {
		printf( " Usage: ./logistic_mpi <batch_size>");
		exit( 0 );
	} else if ( argc == 2 ) {
		batch_size = atoi( argv[1] ); // mini-batch processing
	} else {
		batch_size = INT_MIN; // batch processing
	}

	// initialize/populate mpi specific vars local to each node
	int  numtasks, taskid, len, dest, source;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	MPI_Get_processor_name(hostname, &len);


	/* DATA PREPROCESSING */
	// define data directory for each node
	std::string taskstr = std::to_string( taskid );
	std::string datafile = "./data/data";
	datafile += taskstr + "/train" + taskstr + ".tsv";

	// determine number of instances
	count_instances( datafile );

	// determine number of features
	count_features( datafile, taskid );


	/* DATA INITIALIZATION */
    m = 20; // TEMPORARY TESTING VALUE
    printf( "m = %lu n = %lu\n", m, n );
	Mat X( m, n );
	Vec labels( m );
	double feat_val, label;
	std::ifstream data( datafile );
	for ( ProbSize i=0; i<m; ++i ) {
		for ( ProbSize j=0; j<n; ++j ) {
			data >> feat_val;
			X(i,j) = feat_val;
		}
		data >> label;
		labels[i] = label;
	}
    // std::cout << X << "\n" << labels << "\n";

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
	MPI_Op op;
	MPI_Op_create( (MPI_User_function *)reduce_unique_labels, 1, &op );
	MPI_Allreduce( unique_labels, global_unique_labels, max_size, MPI_INT, op, MPI_COMM_WORLD );
	
	// update local classmap
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
	LogisticRegression clf( n, numlabels );

	// initialize and communicate paramters
	if (taskid == MASTER) {
		// init and send parameters

	} else {
		// recieve network parameters and update local classifier

	}


	/* OPTIMIZATION */



	// perform prediction and model storage tasks on a single node
	if (taskid == MASTER) {
		/* PREDICTION */

		// predict on validation set

		// output prediction results


		/* MODEL STORAGE */

		// store parameters
	} 

	MPI_Finalize();
}
