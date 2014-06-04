#include <iostream>
#include <math.h>
#include "vect.h"
#include <omp.h>

using namespace std;

//tests refraction


Flt test_surface(Flt x, Flt y){
  return (x*x+y*y)/10.0+1.0l;  //why minus sign helps???
}

Flt test_plane(Flt x, Flt y){
  //  return -(x*x+y*y)/5.0+13.0;
  //  return 10.0+sin(4.0*x);
    return 10;
}



int main(){

  //  printf("output: %d \n", apply(test,5);

#define WIDTH 5.0
#define DENSITY 10.0


  //#pragma omp parallel for
  for(int i=-(WIDTH*DENSITY)/2;i<=(WIDTH*DENSITY)/2;i++){
    Ray initial_ray;
    Ray output_ray;

    //create grid of rays pointing upward
    initial_ray.P[0]=((float)i)*(1.0/DENSITY);
    initial_ray.P[1]=0.0;
    initial_ray.P[2]=0.0;
    initial_ray.D[0]=0.0;
    initial_ray.D[1]=0.0;
    initial_ray.D[2]=1.0;
    output_ray.P[0]=0.0;//for some reason error when not initialized. bizzare
    output_ray.P[1]=0.0;
    output_ray.P[2]=0.0;
    output_ray.D[0]=0.5;
    output_ray.D[1]=0.0;
    output_ray.D[2]=-0.5;

#define n_1 1.0
#define n_2 1.05

    if(RefractRay( &initial_ray, &output_ray, n_1, n_2, -5000.0, 5000.0, test_surface)){
      //      cerr << "Refracted ray i=" << i << "\n";
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";

      //      cout << output_ray.P << "\n";
      //      VecMult(5, output_ray.D, Tmp);
      //      VecAdd(output_ray.P,Tmp,Fin);
      //      cout << Fin << "\n\n";

    } else {
      //            cerr << "ERROR: DID NOT Refract ray i=" << i << "\n";
    }

    VecCopy( output_ray.P, initial_ray.P);
    VecCopy( output_ray.D, initial_ray.D);



    if(RefractRay( &initial_ray, &output_ray, n_2, n_1, -5.0, 5000.0, test_plane)){
      //      cerr << "Refracted ray i=" << i << "\n";
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";

      cout << output_ray.P << "\n";
      VecMult(100, output_ray.D, Tmp);
      VecAdd(output_ray.P,Tmp,Fin);
      cout << Fin << "\n\n";

    } else {
      //            cerr << "ERROR: DID NOT Refract ray i=" << i << "\n";
      //	    cout << "###############################" << "\n";
    }


  }
}
