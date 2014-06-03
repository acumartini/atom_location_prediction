all: neuralnet_mpi logistic_mpi

neuralnet_mpi: neuralnet_mpi.cpp
	mpic++ -g -std=c++11 -o neuralnet_mpi neuralnet_mpi.cpp -Wall -Wextra

logistic_mpi: logistic_mpi.cpp
	mpic++ -g -std=c++11 -o logistic_mpi logistic_mpi.cpp -Wall -Wextra

clean:
	rm -f *.o neuralnet_mpi	logistic_mpi
.PHONY: clean
