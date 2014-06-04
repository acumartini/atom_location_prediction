#include <iostream>
#include <math.h>
#include "vect.h"

using namespace std;

//tests using the Calculate Normal function


#define NUM_OF_VECTS 100
//#define URAND() (rand()/(Flt)RAND_MAX)
//#define PI 3.14159265
//#define E  2.71

#define BASE_STEP_SIZE 0.01
//#define TOLERANCE 0.1
//#define MAX_STEPS 10000000



int test(int x){
  return x+1;
}


int apply( int (*f)(int), int x){
  return (*f)(x);
}


Flt test_surface(Flt x, Flt y){
  //  return 10;
  return (x*x+y*y); //z
}



int main(){

  //  printf("output: %d \n", apply(test,5);

  Vec A;
  Vec B, C, normal;
  Vec displace;
  Vec D;

  int i;
  for(i=-20;i<=20;i++){
    A[0]=i*0.1;
    A[1]=0.0;
    A[2]=0.0;
    B[0]=i*0.1;
    B[1]=0.0;
    B[2]=test_surface(B[0],B[1]);
    
    VecSub(B,A,displace);

    CalculateNormal( B, test_surface, normal);
    SnellsLaw( 1.0, 2.0, normal, displace, C);
    VecAdd(B,C,D);
    cout << B << "\n";
    cout << D << "\n\n";
  }
}
