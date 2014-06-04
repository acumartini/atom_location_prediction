#include "mpi.h"
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

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include "Eigen/Core"
using namespace Eigen;

#define  MASTER		0

typedef unsigned long ProbSize;

// globla model variables
std::string datadir;
ProbSize m, n; // numbers of instance and features


void count_instances( int taskid ) {
	struct dirent *pDirent;
	DIR *pDir;
	m = 0;
	pDir = opendir( datadir.c_str() );

	if (pDir != NULL) {
	    while ( ( pDirent = readdir( pDir ) ) != NULL) { m++; }
		m -= 2; // ignore "." and ".." directory reads
	    closedir (pDir);
	}
}

void count_features( int taskid ) {
	std::ifstream infile( datadir );
	std::string line;
	std::getline( infile, line );
    std::istringstream iss( line );
    std::vector<std::string> tokens{
    	std::istream_iterator<std::string>{iss},
    	std::istream_iterator<std::string>{}
    };
    n = tokens.size() - 1; // -1 for label
    printf( "n = %lu\n", n );
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
	std::string datadir = "/data/data";
	datadir += std::to_string( taskid );

	// determine number of instances
	count_instances( taskid );

	// determine number of features
	count_features( taskid );


	/* DATA INITIALIZATION */
	// X = 

        
    /* CLASSIFICATION MODEL INITIALIZATION */


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
