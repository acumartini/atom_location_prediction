#include <iostream>
#include <stdlib.h>
#include "vect.h"

#define NUM_OF_VECTS 100
#define URAND() (rand()/(Flt)RAND_MAX)


using namespace std;

Flt test_surface1(Flt x, Flt y){
  //  return 10;
  return (x*x+y*y)+10.0; //z
}

Flt test_surface2(Flt x, Flt y){
  //  return 10;
  return 20.0; //z
}

Flt test_surface3(Flt x, Flt y){
  //  return 10;
  return 30.0; //z
}



int main(int argc, char ** argv)
{
  Ray rays[NUM_OF_VECTS];
  Ray rays2[NUM_OF_VECTS];//secondary rays

  for(int i=0; i<NUM_OF_VECTS; i++){
    rays[i].P[0]=0;
    rays[i].P[1]=0;
    rays[i].P[2]=0;
    AngleToVec(1.0,URAND()*2*PI,URAND()*PI,(rays[i].D));
  }
  /*
  for(int i=0; i<NUM_OF_VECTS; i++){
    cout << rays[i] << "\n";
  }
  */

    Vec B;
  for(Flt i=0; i<1000; i++){

    RayPoint(rays[3],i,B);
    //    cout << B << "\n";
  }

    Flt error=100.0;
    Flt t=0.0;
    Flt step_size=BASE_STEP_SIZE;
    Flt error_min=100.0;

  for(int ray=0; ray<NUM_OF_VECTS; ray++){
    error = 100.0;
    t = 0.0;
    error_min = 100.0;    
    B[0]=0.0;
    B[1]=0.0;
    B[2]=0.0;

    while(error > TOLERANCE){
      RayPoint(rays[ray],t,B);
      error = test_surface1(B[0],B[1])-B[2];
      if(error < error_min){ error_min = error;}
      t += step_size;
      //      cout << "error: " << error << "\n";
      if(t > BASE_STEP_SIZE * MAX_STEPS){
	//	cout << "out of steps, error: " << error << "\n";
	//	cout << "          error_min: " << error_min << "\n";
	break;
      }
    }

    if(t > BASE_STEP_SIZE * MAX_STEPS){
      cout << "out of steps, error: " << error << "\n";
      cout << "          error_min: " << error_min << "\n";
    } else {
      cout << ray << " best guess:" << B << "\n";
    }


  }

  //I really only care about the number of rays that hit the screen in the end, and I don't actually NEED to define variables for each ray. this is because I can basically do a "spawn" on a ray, and it will start going through it's whole routine.
  // I need a fire_ray() function that takes it's initial conditions and then delivers me a point on the screen (or no point). 
  // This has a bad data structure, and so for speed, I'll instead have a fire_rays() function, that will fire a bunch or rays, given all their initial conditions. Then I'll have some level of cache coherency.
  //I can then parallelize over fire_rays(), and optimize based on the number of rays per fire_rays() call.



}


//returns if the ray ultimately collided with the target
bool castRay(Ray initial_ray, Point p1, Point p2, Point p3){

  Vec vect_intersect, vect_normal;
  Ray output_ray;
  B[0]=0.0;
  B[1]=0.0;
  B[2]=0.0;


  //could turn this into a function... START
  if(!RayIntersection(initial_ray, test_surface1, 0.0, 0.0, vect_intersect)){
    return false;
  }
  VecCopy(vect_intersect,Ray.P);
  CalculateNormal(vect_intersect,test_surface1,vect_normal);
  SnellsLaw(1.0,1.5,vect_normal,B,ray.D);

  if(!RayIntersection(initial_ray, test_surface1, 0.0, 0.0, vect_intersect)){
    return false;
  }
  VecCopy(vect_intersect,Ray.P);
  CalculateNormal(vect_intersect,test_surface1,vect_normal);
  SnellsLaw(1.5,1.0,vect_normal,B,ray.D);
  //...END





}


void TestVectOps(){
    Vec A = {1, 2, 3};
    Vec B = {2, 3, 4};
    Vec C;

    cout << "Starting vectors:\n";
    cout << "A: " << A << "\n";
    cout << "B: " << B << "\n";
    cout << "C: " << C << "\n";

    cout << "VecLen(A):" << VecLen(A) << "\n";
    cout << "VecDot(A,B):" << VecDot(A,B) << "\n";

    cout << "VecCopy(A,C):\n"; 
    VecCopy(A,C);
    cout << "A: " << A << "\n";
    cout << "B: " << B << "\n";
    cout << "C: " << C << "\n";
    cout << "\n";

    cout << "VecAdd(A,B,C):" << "\n";
    VecAdd(A,B,C);
    cout << "A: " << A << "\n";
    cout << "B: " << B << "\n";
    cout << "C: " << C << "\n";
    cout << "\n";

    cout << "VecCross(A,B,C):" << "\n";
    VecCross(A,B,C);
    cout << "A: " << A << "\n";
    cout << "B: " << B << "\n";
    cout << "C: " << C << "\n";
    cout << "\n";

    cout << "VecSub(A,B,C):" << "\n";
    VecSub(A,B,C);
    cout << "A: " << A << "\n";
    cout << "B: " << B << "\n";
    cout << "C: " << C << "\n";
    cout << "\n";

    cout << "VecComb(1.0,A,2.0,B,C):" << "\n";
    VecComb(1.0,A,2.0,B,C);
    cout << "A: " << A << "\n";
    cout << "B: " << B << "\n";
    cout << "C: " << C << "\n";
    cout << "\n";

    cout << "VecUnit(A,B):" << VecUnit(A,B) << "\n";
    cout << "A: " << A << "\n";
    cout << "B: " << B << "\n";
    cout << "C: " << C << "\n";
    cout << "\n";
}
