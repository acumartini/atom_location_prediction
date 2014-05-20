/*
 * logistic.h
 *
 *  Created on: 5-17-2014
 *      Author: martini
 */

#ifndef LOGISTIC_H_
#define LOGISTIC_H_

#include <stdio.h>

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include "Eigen/Core"


typedef unsigned int LayerSize;
typedef Eigen::MatrixXd Mat;
typedef Eigen::Map<Mat> MatMap;
typedef Eigen::RowVectorXd Vec;
typedef Eigen::Map<Vec> VecMap;
typedef Eigen::MatrixXf::Index PredMat;


/* ERROR HANDLING */
#define NO_UPDATE 0

class LogisticRegressionError: public std::exception {
public:
	LogisticRegressionError ( int e ): code( e ) {}
	virtual const char* what() const throw() {
	    if ( code == NO_UPDATE ) {
	    	return "No update for delta (gradient) vector available.";
	    }
	    return "General Error";
	}
private:
	int code;
};


/*
 * Multi-class Logistic Regression Class

 * The logistic regression is fully described by a weight matrix W
 * and bias vector b. Classification is done by projecting data
 * points onto a set of hyperplanes, the distance to which is used to
 * determine a class membership probability.
 */
class LogisticRegression {
public:
	LogisticRegression ( LayerSize n_in, LayerSize n_out, bool distrib=false ) :
		// initialize theta = (W,b) with 0s; W gets the shape (n_in, n_out),
        // while b is a vector of n_out elements, making theta a vector of
        // n_in*n_out + n_out elements
        theta( Vec::Zero( n_in * n_out + n_out ) ),

		// map the weight matrix W to theta using shape = (n_in, n_out)
		W( MatMap( theta.data(), n_in, n_out ) ),

		// map the bias vector b to the last n_out elements of theta
		b( VecMap( theta.data() + n_in * n_out, n_out ) ),

		// intialize delta (gradient) vector and corresponding Maps
		delta( Vec::Zero( n_in * n_out + n_out ) ),
		dW( MatMap( delta.data(), n_in, n_out ) ),
		db( VecMap( delta.data() + n_in * n_out, n_out ) ),

		distributed( distrib ), // controls parameter update behavior

		updated( false ) // prevent reading unitialized variables
	{}
	~LogisticRegression () {}


	/* OPTIMIZATION */
	void compute_gradient ( const Mat& X, const Mat& y ) {
		// compute P( y | X )
		Mat probas = softmax( (X * W).rowwise() + b );

		// compute the error
		Mat error = probas - y;

		// check if the algorithms is used in a distributed setting and only normalize
		// the gradient if running on a single process
		if ( distributed ) { 
			dW = X.transpose() * error;
			db = error.colwise().sum();
		} else {
			// printf( "enter\n" );
			dW = X.transpose() * error;
			// std::cout << dW << std::endl;
			std::cout << dW.norm() << std::endl;
			//printf( "1\n" );
			dW.noalias() += ( W * lambda ) / X.rows(); // apply regularization
			//std::cout << dW << std::endl;
			//printf( "2\n" );
			//std::cout << error << std::endl;
			db = error.colwise().mean();// / X.rows();
			// std::cout << db << std::endl;
			// printf( "exit\n" );
		}

		updated = true;  // now safe to access gradient update data
	}

	Mat softmax ( Mat z ) {
		return ( 1.0 / ( 1.0 + (-z).array().exp() ) );
	}

	void set_theta ( const Vec& theta_update ) { theta << theta_update; }

	void set_parameters ( float a, float l, float e ) {
		alpha = a; lambda = l; epsilon = e;
	}

	const Vec& get_delta () const { 
		if ( !updated ) { throw LogisticRegressionError( NO_UPDATE ); }
		else {
			return delta; 
		}
	}

	bool converged () { return dW.norm() <= epsilon; }

	void update_theta () {
		if ( !updated ) { throw LogisticRegressionError( NO_UPDATE ); } 
		else {
			// std::cout << delta << std::endl;
			theta.noalias() -= alpha * delta;
		}
	}


	/* PREDICTION AND ERROR */
	Mat predict_proba ( Mat& X ) {
		return softmax( (X * W).rowwise() + b );
	}

	Vec predict ( Mat& X ) {
		PredMat pred[X.rows()];
		Mat probas = softmax( (X * W).rowwise() + b );
		for ( int i=0; i < probas.rows(); ++i ) {			
			probas.row( i ).maxCoeff( &pred[i] );
		}
		// convert to vector
		Vec predvec( X.rows() );
		for ( int i=0; i < probas.rows(); ++i ) {			
			predvec[i] = pred[i];
		}
		return predvec;
	}

	// TODO: float errors () {}


private:
	// model parameters
	Vec theta, delta;
	MatMap W, dW;
	VecMap b, db; 
	bool distributed, updated;

	// update parameters
	float alpha = 0.9, lambda = 0.0, epsilon = 0.00001;
};




#endif /* LOGISTIC_H_ */