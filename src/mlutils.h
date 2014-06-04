/*
 * mlutils.h
 *
 *  Created on: 5-17-2014
 *      Author: martini
 */

#ifndef MLUTILS_H_
#define MLUTILS_H_

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include "logistic.h"  // just to get typdefs (fix this later)

typedef std::unordered_map<double, int> ClassMap;
typedef std::unordered_set<int> ClassSet;

namespace mlu {
	
	void scale_features ( Mat& X, int new_max, int new_min ) {
		Vec X_min = X.colwise().minCoeff();
		Vec X_max = X.colwise().maxCoeff();
		Mat tmp = X.rowwise() - X_min;
		Vec tmp2 = (X_max - X_min) * (new_max - new_min);
		tmp = tmp * tmp2.asDiagonal().inverse();
		tmp = tmp.array() + new_min;
		X = tmp.matrix();
	}

	void get_unique_labels( Vec& y, ClassMap& unique ) {
		ClassMap::const_iterator got;
		
		// find unique items and map them to incremental indices
		unsigned int index_count = 0;
		for ( int i=0; i<y.size(); ++i ) {
			got = unique.find( y[i] );
			if ( got == unique.end() ) {  // new unique
				unique.emplace( y[i], index_count++ );
			}
		}
		// for ( auto& kv : unique ) {
		// 	printf( "k = %lf, v = %d\n", kv.first, kv.second );
		// }
	}

	Mat format_labels ( Vec& y, ClassMap& unique ) {
		// create label matrix
		if ( unique.size() == 2 ) { // simple classification problem (column vector)
			Mat labels = Mat::Zero( y.size(), 1 );
			for ( int i=0; i< y.size(); ++i ) {
				labels( i, 0 ) = unique[y[i]];
			}
			return labels;
		} else { // multiclass classification (one-vs-all matrix)
			Mat labels = Mat::Zero( y.size(), unique.size() );
			for ( int i=0; i< y.size(); ++i ) {
				labels( i, unique[y[i]] ) = 1.0;
			}
			return labels;
		}
	}

	Mat confusion_matrix( const Mat& y, const Vec& pred ) {
		if ( y.cols() == 1 ) {
			Mat cm = Mat::Zero( 2, 2 );
			int y_index;
			for ( int i = 0; i < pred.size(); ++i ) {
				y_index = y(i,0);
				cm( y_index, pred[i] ) += 1;
			}
			return cm;
		} else {
			Mat cm = Mat::Zero( y.cols(), y.cols() );
			Eigen::MatrixXf::Index   y_index;
			for ( int i = 0; i < pred.size(); ++i ) {
				y.row(i).maxCoeff( &y_index );
				cm( y_index, pred[i] ) += 1;
			}
			return cm;
		}
	}

}

#endif /* MLUTILS_H_ */
