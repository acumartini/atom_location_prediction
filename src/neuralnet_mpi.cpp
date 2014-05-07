#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include "Eigen/Core"

#define  MASTER		0
#define  TAG_0       0

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
    long offset;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	MPI_Get_processor_name(hostname, &len);


	/***** Master task only ******/

	// perform data preprocessing based on number of workers and batch_size
	if (taskid == MASTER) {
		printf( "MASTER: Number of MPI tasks is: %d\n",numtasks );

		/* DATA PREPROCESSING */

		printf( "Data preprocessing from MASTER task %d on %s!\n", taskid, hostname );
		// partition data
		// TODO: partition vector into chunks and send each task its share
		const long datasize = 16;
        const long numfeats = 4; // to be populated while loading dataset
        const long numlabels = 1; // for testing binary classification only
        MatrixXf data = MatrixXf::Zero( datasize, numfeats );
        float min[numfeats]; // stores the overall min for each column
        float max[numfeats]; // stores the overall max for each column
        float labels[datasize * numlabels];

        // Load data and labels into arrays from file
        // TEMP: populate fictitious dataset
        // NOTE: record max and min for each column in the max/min arrays
        for ( long i=0; i<datasize; ++i ) {
            for ( long j=0; j<numfeats; ++j ) {
            	long index = (i * numfeats) + j;
                data(i, j) = 10.0 + j;
            }
            labels[i] = 1.0;
        }

        // Shuffle data to randomize instances

        // Scale features
        // TODO: use max/min arrays to scale data to between -1 and 1
	        // # scales all features in dataset X to values between new_min and new_max
			// X_min, X_max = X.min(0), X.max(0)
			// return (((X - X_min) / (X_max - X_min + 0.000001)) * (new_max - new_min)) + new_min


        /* DATA MARSHALLING */

        long chunksize = datasize / numtasks;
		offset = chunksize;
		for (dest=1; dest<numtasks; dest++) {
			MPI_Send( &offset, 1, MPI_LONG, dest, TAG_0, MPI_COMM_WORLD );
			MPI_Send( &chunksize, 1, MPI_LONG, dest, TAG_0, MPI_COMM_WORLD );
			MPI_Send( &numfeats, 1, MPI_LONG, dest, TAG_0, MPI_COMM_WORLD );
			MPI_Send( &numlabels, 1, MPI_LONG, dest, TAG_0, MPI_COMM_WORLD );
			MPI_Send( data.data()[offset], chunksize * numfeats, MPI_FLOAT, dest, TAG_0, MPI_COMM_WORLD );
			MPI_Send( &labels[offset], chunksize, MPI_FLOAT, dest, TAG_0, MPI_COMM_WORLD );
			printf( "Sent %ld elements to task %d offset= %ld\n", chunksize, dest, offset );
			offset += chunksize;
		}

	    MatrixXf X = MatrixXf::Zero(2, 2);
		MPI_Send( X.data(), X.size(), MPI_FLOAT, 1, TAG_0, MPI_COMM_WORLD );


		/* CLASSIFICATION MODEL INITIALIZATION *

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

	/***** Non-master tasks only *****/

	if (taskid > MASTER) {
		printf ("Hello from task %d on %s!\n", taskid, hostname);

		/* DATA INITIALIZATION */

		// variables for temporary message storage
		long chunksize_msg, numfeats_msg, numlabels_msg;
		source = MASTER;
		
		// recieve data partition
		MPI_Recv( &offset, 1, MPI_LONG, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		printf( "Task %d offset = %ld\n", taskid, offset );
		MPI_Recv( &chunksize_msg, 1, MPI_LONG, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		printf( "Task %d chunksize_msg = %ld\n", taskid, chunksize_msg );
		MPI_Recv( &numfeats_msg, 1, MPI_LONG, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		printf( "Task %d numfeats_msg = %ld\n", taskid, numfeats_msg );
		MPI_Recv( &numlabels_msg, 1, MPI_LONG, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
		printf( "Task %d numlabels_msg = %ld\n", taskid, numlabels_msg );
        
        // initialize local data storage
        const long chunksize = chunksize_msg;
        const long numfeats = numfeats_msg;
        const long numlabels = numlabels_msg;
        MatrixXf data = MatrixXf::Zero( chunksize, numfeats );
        float labels[chunksize * numlabels];

        // receive data and labels
		MPI_Recv( data.data(), chunksize * numfeats, MPI_FLOAT, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
        //printf( "Task %d data[0] = %f\n", taskid, data[0] );
        //printf( "Task %d data[chunksize * numfeats-1] = %f\n", taskid, data[chunksize*numfeats-1] );
		std::cout << "data:\n" << data << std::endl;
		MPI_Recv( &labels, chunksize, MPI_FLOAT, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );
        printf( "Task %d labels[0] = %f\n", taskid, labels[0] );

        // convert data to Matrix objects (i.e., Eigen Matrices)
        MatrixXf X = MatrixXf::Zero(chunksize, numfeats);
        for ( int i=0; i<chunksize; ++i ) {
            for ( int j=0; j<numfeats; ++j ) {
                X(i,j) = data[(i*numfeats) + j];
            }
        }
        MatrixXf y = MatrixXf::Zero(chunksize, numlabels);
        for ( int i=0; i<chunksize; ++i ) {
            for ( int j=0; j<numlabels; ++j ) {
                y(i,j) = labels[(i*numlabels) + j];
            }
        }

        std::cout << "X:\n" << X << std::endl;
        std::cout << "y:\n" << y << std::endl;
        std::cout << "X * y:\n" << X * y << std::endl;
        
        if (taskid == 1) {
            X = MatrixXf::Zero(2, 2);
            MPI_Recv( X.data(), X.size(), MPI_FLOAT, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status );

            std::cout << "Eigen from MASTER:\n" << X << std::endl;
        } 
        
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
