#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include <string>
#include <dirent.h>
#include <sstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include "Eigen/Core"
using namespace Eigen;

#define  MASTER		0

typedef unsigned long ProbSize;

// globla model variables
ProbSize m, n; // numbers of instance and features


void count_instances( int taskid ) {
	struct *pDirent;
	DIR *pDir;
	m = 0;

	pDir = opendir( "/data/part" + std::to_string( taskid ) );
	if (pDir != NULL) {
	    while ( ( pDirent = readdir( pDir ) ) != NULL) { m++; }
		m -= 2; // remove "." and ".." directory reads
	    closedir (pDir);
	}
}

void count_features( int taskid ) {
	std::string line;
	std::getline(infile, line);
    istringstream iss( line );
    vector<string> tokens{istream_iterator<string>{iss},
         istream_iterator<string>{}};
    n = tokens.size() - 1; // -1 for label
    printf( "n = %lu\n", n );
}

int main (int argc, char *argv[]) {
    // handle cmd args
	int batch_size;
	if ( argc > 1 ) {
		printf( " Usage: ./logistic_mpi <batch_size>");
		exit( 0 );
	} else if ( argc == 1 ) {
		batch_size = atoi( argv[0] ); // mini-batch processing
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
	// determine number of instances
	count_instances( taskid );

	// determine number of features
	count_features( taskid );


	/* DATA INITIALIZATION */
	// X = 

        
    /* CLASSIFICATION MODEL INITIALIZATION */


	// initialize and communicate paramters
	if (taskid == MASTER) {

	} else {
		// recieve network parameters and pass to local classifier

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
