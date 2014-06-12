all: neuralnet_mpi logistic_mpi LR_predict

neuralnet_mpi: neuralnet_mpi.cpp
	mpic++ -std=c++11 -O2 -o neuralnet_mpi neuralnet_mpi.cpp -Wall -Wextra

logistic_mpi: logistic_mpi.cpp
	mpic++ -std=c++11 -O2 -o logistic_mpi logistic_mpi.cpp -Wall -Wextra

LR_predict: LR_predict.cpp
	g++ -std=c++11 -O2 -o LR_predict LR_predict.cpp -Wall -Wextra

clean:
	rm -f *.o neuralnet_mpi	logistic_mpi LR_predict
.PHONY: clean
