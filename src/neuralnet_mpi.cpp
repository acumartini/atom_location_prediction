#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits.h>

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include "Eigen/Core"
// USING_PART_OF_NAMESPACE_EIGEN

#define  MASTER		0
#define  TAG_0      0

using namespace Eigen;


int main (int argc, char *argv[]) {
    // handle cmd args
	// TODO: add <hidden_layer_sizes> (i.e., "100-100-50") argument processing
	int batch_size;
	if ( argc > 1 ) {
		printf( " Usage: ./neuralnet_mpi <batch_size>");
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


	/***** MASTER TASK ONLY ******/

	// perform data preprocessing based on number of workers and batch_size
	if (taskid == MASTER) {
		printf( "MASTER: Number of MPI tasks is: %d\n",numtasks );

		/* DATA PREPROCESSING */

		// Load dataset
        // TEMP: populate fictitious dataset
        // NOTE: record max and min for each column in the max/min arrays
		long datasize = 16;
        long numfeats = 4; // to be populated while loading dataset
        long numlabels = 1; // for testing binary classification only
        
        //float min[numfeats]; // stores the overall min for each column
        //float max[numfeats]; // stores the overall max for each column
        MatrixXf data = MatrixXf( datasize, numfeats ); // row-major order!
        VectorXf labels_vec = VectorXf( datasize );
        
        for ( long i=0; i<datasize; ++i ) {
            for ( long j=0; j<numfeats; ++j ) {
                data(i, j) = i + j;
            }
            labels_vec[i] = 1.0; // populate a vector with labels for each instance
        }
        data(0,0) = 100.1;
        data(1,0) = 200.1;
        data(0,1) = 300.1;

        // Shuffle data/labels to randomize instances
        
        // Reformat labels for multi-class classification
        MatrixXf labels = MatrixXf::Zero( datasize, numlabels );
        // TODO: create class_map from unique classes with k:v = <true_label>:<column_index> 
        for ( long i=0; i<datasize; ++i ) {
            // labels( i, class_map[y_vec[i]] ) = 1.0;
        }

        // Scale features
        // TODO: use max/min arrays to scale data to between -1 and 1
	        // # scales all features in dataset X to values between new_min and new_max
			// X_min, X_max = X.min(0), X.max(0)
			// return (((X - X_min) / (X_max - X_min + 0.000001)) * (new_max - new_min)) + new_min


        /* DATA MARSHALLING */

        long chunksize = datasize / numtasks;
        
        // load MASTER data
        MatrixXf X = MatrixXf( chunksize, numfeats );
        MatrixXf y = MatrixXf( chunksize, numlabels );
        memcpy( X.data(), data.data(), chunksize * numfeats * sizeof(float) ); 
        memcpy( y.data(), labels.data(), chunksize * numlabels *  sizeof(float) );
        std::cout << "MASTER X:\n" << X << std::endl;
        std::cout << "MASTER y:\n" << y << std::endl;
        
        // send data to workers
        long offset = chunksize;
		for (dest=1; dest<numtasks; dest++) {
			MPI_Send( &chunksize, 1, MPI_LONG, dest, TAG_0, MPI_COMM_WORLD );
			MPI_Send( &numfeats, 1, MPI_LONG, dest, TAG_0, MPI_COMM_WORLD );
			MPI_Send( &numlabels, 1, MPI_LONG, dest, TAG_0, MPI_COMM_WORLD );
			MPI_Send( data.data() + offset * numfeats, chunksize * numfeats, MPI_FLOAT, dest, TAG_0, MPI_COMM_WORLD );
			MPI_Send( labels.data() + offset * numlabels, chunksize * numlabels, MPI_FLOAT, dest, TAG_0, MPI_COMM_WORLD );
			printf( "Sent %ld instances to task %d offset= %ld\n", chunksize, dest, offset );
			offset += chunksize;
		}


		/* CLASSIFICATION MODEL INITIALIZATION */

		// pass network structure and processing parameters message

		// initialze MASTER NN

		// initialize network parameters
		// TODO: create randomized set of parameters stored in contiguous memory
		// to be packed and unpacked as needed.  The important part of doing this on
		// the MASTER first is that all networks will start with the same parameter set.

		// set MASTER NN parameters


		/* OPTIMIZATION */

		// optimize
		offset = 0;


		/* PREDICTION */

		// predict on validation set

		// output prediction results


		/* MODEL STORAGE */

		// store parameters
	} 

	/***** NON-MASTER TASKS ONLY *****/

	if (taskid > MASTER) {
		printf ("Hello from task %d on %s!\n", taskid, hostname);

		/* DATA INITIALIZATION */

		long chunksize, numfeats, numlabels;
		source = MASTER;
		
		// recieve data partition
		MPI_Recv( &chunksize, 1, MPI_LONG, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		MPI_Recv( &numfeats, 1, MPI_LONG, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		MPI_Recv( &numlabels, 1, MPI_LONG, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		printf( "Task %d chunksize = %ld\n", taskid, chunksize );
		printf( "Task %d numfeats = %ld\n", taskid, numfeats );
		printf( "Task %d numlabels = %ld\n", taskid, numlabels );
        
        // initialize local data storage
        MatrixXf X = MatrixXf( chunksize, numfeats );
        MatrixXf y = MatrixXf( chunksize, numlabels );

        // receive data and labels
		MPI_Recv( X.data(), chunksize * numfeats, MPI_FLOAT, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		MPI_Recv( y.data(), chunksize * numlabels, MPI_FLOAT, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		std::cout << "task " << taskid << " X:\n" << X << std::endl;
        std::cout << "task " << taskid << " y:\n" << y << std::endl;
        
        
        /* CLASSIFICATION MODEL INITIALIZATION */

		// recieve network structure and processing paramters info

		// initialize local neuralnet_openmp instance
		// TODO: each NN instance is set with identical structure and processing parameters

		// recieve network parameters and pass to local instance
		// TODO: each NN instance gets the same set of randomized parameters


		/* OPTIMIZATION */

		// optimize

	}

	MPI_Finalize();
}
