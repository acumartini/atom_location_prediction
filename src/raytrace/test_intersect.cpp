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

//#define MAX_STEPS 10000000



int test(int x){
  return x+1;
}


int apply( int (*f)(int), int x){
  return (*f)(x);
}


Flt test_surface(Flt x, Flt y){
  //  return 10;
  return (x*x+y*y)+1; //z
}



int main(){

  //  printf("output: %d \n", apply(test,5);

  Ray ray;
  ray.P[0]=0;
  ray.P[1]=0;
  ray.P[2]=0;

  int i;
  for(i=-20;i<=20;i++){
    Vec Tmp,D;
    Flt tstart = -10.0;
    Flt tend = 10.0;
    Flt t = 0.0;
    AngleToVec(1.0l, 0.0l, PI/(20.0*20.0)*i, ray.D);

    RayIntersection(&ray, test_surface, tstart, tend, Tmp);

    //changed to sub for some reasond?
    VecAdd(ray.P,Tmp,D);

    cout << ray.P << "\n";
    cout << D << "\n\n";
  }
}
