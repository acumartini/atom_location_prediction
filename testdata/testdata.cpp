#include <iostream>
#include <stdlib.h>
#include <math.h>

using namespace std;

#define PI 3.14
#define E 2.71
#define GAUSS(x,y,x0,y0,sigma) (1/(sigma*sqrt(2*PI))*pow(E,-(pow(x-x0,2.0)+pow(y-y0,2.0))/(2*pow(sigma,2.0))))

int main(int argv, char ** args){

  if( argv != 6 ){
    cout << "Usage: " << args[0] << " ROWS COLS ATOM_X ATOM_Y SIGMA\n";
    return 0;
  }
    

  int ROWS  = atoi(args[1]);
  int COLS  = atoi(args[2]);
  int X0    = atoi(args[3]);
  int Y0    = atoi(args[4]);
  int sigma = atoi(args[5]);

  cout << "#";
  cout << " ROWS:" << ROWS;
  cout << " COLS:" << COLS;
  cout << " X0:"   << X0;
  cout << " Y0:"   << Y0;
  cout << " sigma:"<< sigma;
  cout << "\n"; 
  cout << "ROWS=" << ROWS << "\n";
  cout << "COLS=" << COLS << "\n"; 

  for(int y=0;y<ROWS;y++){
    for(int x=0;x<COLS;x++){
      cout << GAUSS(x,y,X0,Y0,sigma) << "\n";
    }
  }

}
