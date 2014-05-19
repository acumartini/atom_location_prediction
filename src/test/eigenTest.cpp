#include <iostream>
#include "../Eigen/Dense"
#include "../Eigen/Core"

//#include <Eigen/Core>
// #include "../Eigen/Array"

//USING_PART_OF_NAMESPACE_EIGEN

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EigenMatrixDbl;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixDblRowMajor;

using namespace Eigen;

int main()
{
     //  MatrixXd m(2,2);
     //    m(0,0) = 3;
     //      m(1,0) = 2.5;
     //        m(0,1) = -1;
     //          m(1,1) = m(1,0) + m(0,1);
     //            std::cout << m << std::endl;

     // std::cout << std::endl;

     Matrix2d a;
     a << 1, 2, 3, 4;

     MatrixXd b(2,1);
     b << 2, 3;

     std::cout << a << std::endl;
     std::cout << b << std::endl;
     std::cout << a * b << std::endl;

     std::cout << std::endl;

     // std::cout << Map<MatrixXd, 0, InnerStride<2> >( a ) << std::endl;
     // std::cout << Map<Vector2d>( b ) << std::endl;

     // std::cout << std::endl;

    EigenMatrixDbl m(3, 4);
    m << 1, 2, 3, 3.5,
         4, 5, 6, 6.5,
         7, 8, 9, 9.5;

    std::cout << m << std::endl << std::endl;


    EigenMatrixDblRowMajor n(3, 4);
    n << m;
    std::cout << n << std::endl << std::endl;

    Eigen::Map<EigenMatrixDblRowMajor> B2 = Eigen::Map<EigenMatrixDblRowMajor>(n.data(),4,3);

    std::cout << B2 + B2 << std::endl << n << std::endl;

    B2 += B2;

    std::cout << B2 << std::endl << n << std::endl;

    std::cout << std::endl;

    std::cout << B2 / 2 << std::endl << n / 5 << std::endl;

    Vector3d v;
    v << 1, 2, 3;
    n.colwise() += v;
    std::cout << n << std::endl;



     return 0;
}
