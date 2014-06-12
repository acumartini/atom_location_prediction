all: neuralnet_mpi logistic_mpi logistic_predict

neuralnet_mpi: neuralnet_mpi.cpp
	mpic++ -std=c++11 -O2 -o neuralnet_mpi neuralnet_mpi.cpp -Wall -Wextra

logistic_mpi: logistic_mpi.cpp
	mpic++ -std=c++11 -O2 -o logistic_mpi logistic_mpi.cpp -Wall -Wextra

logistic_predict: logistic_predict.cpp
	g++ -std=c++11 -O2 -o logistic_predict logistic_predict.cpp -Wall -Wextra

clean:
	rm -f *.o neuralnet_mpi	logistic_mpi logistic_predict
.PHONY: clean
