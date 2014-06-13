all: logistic_mpi logistic_predict

logistic_mpi: logistic_mpi.cpp
	mpic++ -std=c++11 -O2 -o logistic_mpi logistic_mpi.cpp -Wall -Wextra

logistic_predict: logistic_predict.cpp
	g++ -std=c++11 -O2 -o logistic_predict logistic_predict.cpp -Wall -Wextra

clean:
	rm -f *.o logistic_mpi logistic_predict
.PHONY: clean
