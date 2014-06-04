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
  Vec B, C, normal;

  int i;
  for(i=-20;i<=20;i++){
    B[0]=i*0.1;
    B[1]=0;
    B[2]=test_surface(B[0],B[1]);
    CalculateNormal( B, test_surface, normal);
    VecAdd(B,normal,C);
    cout << B << "\n";
    cout << C << "\n\n";
  }
}
