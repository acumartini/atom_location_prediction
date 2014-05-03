#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define  MASTER		0

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
	int  numtasks, taskid, len, dest, source, tag1, tag2, tag3, tag4; 
    long offset, chunksize;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	MPI_Get_processor_name(hostname, &len);

	// msg test data
    float **data; // an array of float arrays to store instance data
    float *labels; // an array of float (to simplify vector math) labels
	tag1 = 1, tag2 = 2, tag3 = 3, tag4 = 4;

	/***** Master task only ******/

	// perform data preprocessing based on number of workers and batch_size
	if (taskid == MASTER) {
		printf( "MASTER: Number of MPI tasks is: %d\n",numtasks );
		printf( "Data preprocessing from MASTER task %d on %s!\n", taskid, hostname );
		// partition data
		// TODO: partition vector into chunks and send each task its share
		const long datasize = 16;
        const long numfeats = 4; // to be populated while loading dataset
        data = new float*[datasize];
        labels = new float[datasize];
        for ( long i=0; i<datasize; ++i ) {
            data[i] = new float[numfeats];
            for ( long j=0; j<numfeats; ++j ) {
                data[i][j] = 0.0 + j;
            }
            labels[i] = 0.0;
        }
		
        chunksize = datasize / numtasks;
		offset = chunksize;
		for (dest=1; dest<numtasks; dest++) {
			MPI_Send(&offset, 1, MPI_LONG, dest, tag1, MPI_COMM_WORLD);
			MPI_Send(&chunksize, 1, MPI_LONG, dest, tag2, MPI_COMM_WORLD);
			MPI_Send( &data[offset], chunksize, MPI_FLOAT, dest, tag3, MPI_COMM_WORLD );
			MPI_Send( &labels[offset], chunksize, MPI_FLOAT, dest, tag4, MPI_COMM_WORLD );
			printf( "Sent %d elements to task %d offset= %d\n", chunksize, dest, offset );
			offset += chunksize;
		}

		// pass network structure and processing parameters message

		// initialze MASTER NN

		// initialize network parameters
		// TODO: create randomized set of parameters stored in contiguous memory
		// to be packed and unpacked as needed.  The important part of doing this on
		// the MASTER first is that all networks will start with the same parameter set.

		// set MASTER NN parameters

		// optimize
		offset = 0;

		// predict on validation set

		// output prediction results

		// store parameters
	} 

	/***** Non-master tasks only *****/

	if (taskid > MASTER) {
		printf ("Hello from task %d on %s!\n", taskid, hostname);
		source = MASTER;
		
		// recieve data partition
		MPI_Recv(&offset, 1, MPI_INT, source, tag1, MPI_COMM_WORLD, &status);
		printf( "Task %d offset = %ld\n", taskid, offset );
		MPI_Recv(&chunksize, 1, MPI_INT, source, tag2, MPI_COMM_WORLD, &status);
		printf( "Task %d chunksize = %ld\n", taskid, chunksize );
        
        // initialize local data storage
        data = new float[chunksize];

		MPI_Recv(&data[offset], chunksize, MPI_FLOAT, source, tag3, MPI_COMM_WORLD, &status);
        printf( "Task %d data[offset] = %f\n", taskid, data[offset] );
		MPI_Recv(&labels[offset], chunksize, MPI_FLOAT, source, tag3, MPI_COMM_WORLD, &status);
        printf( "Task %d data[offset] = %f\n", taskid, data[offset] );

		// recieve network structure and processing paramters info

		// initialize local neuralnet_openmp instance
		// TODO: each NN instance is set with identical structure and processing parameters

		// recieve network parameters and pass to local instance
		// TODO: each NN instance gets the same set of randomized parameters

		// optimize

	}

	MPI_Finalize();
}
