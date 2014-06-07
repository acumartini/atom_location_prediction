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


typedef unsigned long ProbSize;
typedef unsigned int LayerSize;
typedef Eigen::MatrixXd Mat;
typedef Eigen::Map<Mat> MatMap;
typedef Eigen::RowVectorXd Vec;
typedef Eigen::Map<Vec> VecMap;
typedef Eigen::MatrixXf::Index PredMat;


/* ERROR HANDLING */
#define NO_UPDATE 0
#define INVALID_THETA_UPDATE 1
#define INVALID_DELTA_UPDATE 1

class LogisticRegressionError: public std::exception {
public:
	LogisticRegressionError ( int e ): code( e ) {}
	virtual const char* what() const throw() {
	    if ( code == NO_UPDATE ) {
	    	return "No updated delta (gradient) vector available.";
	    } else if ( code == INVALID_THETA_UPDATE ) {
	    	return "Given theta parameters do not match the size of the current "
	    		   "theta parameters.";
	    } else if ( code == INVALID_DELTA_UPDATE ) {
	    	return "Given delta parameters do not match the size of the current "
	    		   "delta parameters.";
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

	/*
	 * Computes a the updated detla (gradient) vector given a set of instances X with 
	 * labels y.
	 *
	 * @params: X - matrix of m instances with n_in features
	 * 			y - matrix of m labels with n_out columns
	 */
	void compute_gradient ( const Mat& X, const Mat& y ) {
		Mat probas = softmax( (X * W).rowwise() + b ); // compute P( y | X )
		Mat error = probas - y; // compute the error

		// check if the algorithm is used in a distributed setting and only normalize
		// the gradient if running on a single process
		if ( distributed ) { 
			dW = X.transpose() * error;
			db = error.colwise().sum();
		} else {
			dW = X.transpose() * error;
			dW.noalias() += ( W * lambda ) / X.rows(); // apply regularization
			db = error.colwise().mean();
			delta.array() /= X.rows();
		}

		updated = true;  // now safe to access gradient update data
	}

	/*
	 * Called if using distributed model to normalize gradient based on total number
	 * of instances.
	 */
	void normalize_gradient( const ProbSize& m ) {
		delta.array() /= m;
	}

	/*
	 * Called if using distributed model to regularize gradient based on total number
	 * of instances.
	 */
	void regularize_gradient( const ProbSize& m ) {
		dW.noalias() += ( W * lambda ) / m; // do not regularize bias updates
	}

	Mat softmax ( Mat z ) {
		return ( 1.0 / ( 1.0 + (-z).array().exp() ) );
	}

	/*
	 * Allows the client to set the theta parameters 
	 */
	void set_theta ( const Vec& theta_update ) { 
		if ( theta_update.size() != theta.size() ) {
			throw LogisticRegressionError( INVALID_THETA_UPDATE );
		}
		theta << theta_update;
	}

	/*
	 * Allows the client to set internal optimization variables.
	 */
	void set_parameters ( float a, float l, float e ) {
		alpha = a; lambda = l; epsilon = e;
	}

	/*
	 * Returns the size of the delta vector.
	 */
	int get_parameter_size () const {
		return theta.size();
	}

	/*
	 * Returns the current delta (gradient) vector.
	 */
	Vec& get_delta () { 
		if ( !updated ) { throw LogisticRegressionError( NO_UPDATE ); }
		else {
			return delta; 
		}
	}

	/*
	 * Allows the client to set the delta parameters 
	 */
	void set_delta ( const Vec& delta_update ) { 
		if ( delta_update.size() != theta.size() ) {
			throw LogisticRegressionError( INVALID_DELTA_UPDATE );
		}
		delta << delta_update;
	}	

	/*
	 * Returns true if the magnitude of the gradient is less than or equal to the convergence
	 * theshold epsilon.
	 *
	 * @params: mag - a double by reference that captures the gradient norm value
	 */
	bool converged ( double& mag ) { 
		mag = dW.norm();
		return mag <= epsilon;
	}

	/*
	 * Updates the current theta parameters using the pre-computed gradient update if avaiable.
	 * Throws an error is no new gradient update is avaible.
	 */
	void update_theta () {
		if ( !updated ) { throw LogisticRegressionError( NO_UPDATE ); } 
		else {
			theta.noalias() -= alpha * delta;
			updated = false; // disallow theta updates until the gradient is re-computed
		}
	}

	/* PREDICTION */
	
	/*
	 * Returns the probability prediction matrix for the instances in X.
	 *
	 * @params: X - matrix of instances with n_in features
	 */
	Mat predict_proba ( Mat& X ) {
		return softmax( (X * W).rowwise() + b );
	}

	/*
	 * Returns a vector of class predictions for the instances in X.
	 *
	 * @params: X - matrix of instances with n_in features
	 */
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
