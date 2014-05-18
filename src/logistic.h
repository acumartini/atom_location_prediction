/*
 * logistic.h
 *
 *  Created on: 5-17-2014
 *      Author: martini
 */

#ifndef LOGISTIC_H_
#define LOGISTIC_H_

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include "Eigen/Core"

using namespace Eigen;

typedef unsigned int LayerSize;

class LogisticRegression {
public:
	LogisticRegression ( LayerSize n_in, LayerSize n_out ) {
		// initialize the weight matrix W with shape = (n_in, n_out)
		W = MatrixXf::Zero( n_in, n_out );

		// initialize the bias vector b with shape = n_out
		b = MatrixXf::Zero( n_out );

		updated = false; // prevent reading unitialized variables
	}
	~LogisticRegression () {}

	void update ( const MatrixXf& X, const MatrixXf& y ) {
		// compute P( y | X )
		probas = softmax( X.dot( W ) + b );

		// compute the error
		error = probas - y;

		// compute the gradient
		dW = X.transpose().dot( error ) / X.rows();

		// compute bias update
		db = ( error.rowwise().sum() );

		updated = true;  // now safe to access model information
	}

	MatrixXf& softmax ( MatrixXf& z ) {
		return ( 1.0 / ( 1.0 + ( -z ).array()exp() ) );
	}

private:
	MatrixXf W, b, probas;
	bool updated;
};

#endif /* LOGISTIC_H_ */