#include <iostream>
#include <math.h>
#include "vect.h"
//#include <omp.h>

using namespace std;

//tests refraction

//a t D
#define p_n   1.453675 //index of refraction
#define p_a   10.0  // 10mm   
#define p_t    5.0  //  5mm
#define p_D    5.0  //  5mm
#define p_C   (-0.128)
#define p_k   (-0.824)
#define p_D0  17.705
//#define p_D2   -0.0211
//#define p_D4   -0.0871
#define p_D2   (0.0211*0.0211) //scetch
#define p_D4   (0.0871*0.0871*0.0871*0.0871)
#define p_Z(r) ((p_C*r*r)/(1.0+sqrt(1.0-(1.0+p_k)*p_C*p_C*r*r))+p_D0+p_D2*r*r+p_D4*r*r*r*r)
//#define p_Z(r) p_D0-r*r/100.0
#define p_CCD_Z 90.0  // 90 mm


#define WIDTH 5.0
#define DENSITY 10.0

/* START SURFACE DEFINITIONS */

#define n_1 1.0

Flt glass1(Flt x, Flt y){
  return p_a+0.01*x;
}

#define n_2 p_n

Flt lens_surface1(Flt x, Flt y){
  return p_Z(sqrt(x*x+y*y));
}

#define n_3 1.0

Flt lens_surface2(Flt x, Flt y){
  return (p_CCD_Z - p_Z(sqrt(x*x+y*y)));
}

#define n_4 p_n

Flt glass2(Flt x, Flt y){
  return (p_CCD_Z - p_a);
}

#define n_5 1.0

Flt image_plane(Flt x, Flt y){
  return p_CCD_Z;
}
/* END SURFACE DEFINITIONS */


int main(){

  //#pragma omp parallel for
  //  for(int i=-(WIDTH*DENSITY)/2;i<=(WIDTH*DENSITY)/2;i++){
  #define theta_range PI/5.0
  #define dtheta theta_range/100.0
  for(int i=-theta_range/(dtheta*2.0); i<theta_range/(dtheta*2.0); i++){

    Ray initial_ray;
    Ray output_ray;

    initial_ray.P[0]=0.0;
    initial_ray.P[1]=0.0;
    initial_ray.P[2]=0.0;
    AngleToVec( 1.0, 0.0, ((float)i)*(dtheta), initial_ray.D);

    //create grid of rays pointing upward
    /*
    initial_ray.P[0]=((float)i)*(1.0/DENSITY);
    initial_ray.P[1]=0.0;
    initial_ray.P[2]=0.0;
    initial_ray.D[0]=0.0;
    initial_ray.D[1]=0.0;
    initial_ray.D[2]=1.0;
    */
    /*    output_ray.P[0]=0.0;//for some reason error when not initialized. bizzare
    output_ray.P[1]=0.0;
    output_ray.P[2]=0.0;
    output_ray.D[0]=0.5;
    output_ray.D[1]=0.0;
    output_ray.D[2]=-0.5;*/


    if(RefractRay( &initial_ray, &output_ray, n_1, n_2, -5000.0, 5000.0, glass1)){
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";
      //   cerr << "Refracted ray i=" << i << "\n";


      /*
      cout << output_ray.P << "\n";
      VecMult(100, output_ray.D, Tmp);
      VecAdd(output_ray.P,Tmp,Fin);
      cout << Fin << "\n\n";
      */

    } else {
         cerr << "ERROR: 1 DID NOT Refract ray i=" << i << "\n";
    }

    VecCopy( output_ray.P, initial_ray.P);
    VecCopy( output_ray.D, initial_ray.D);

    if(RefractRay( &initial_ray, &output_ray, n_2, n_3, -5000.0, 5000.0, lens_surface1)){
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";
      //      cerr << "Refracted ray i=" << i << "\n";
    } else {
                  cerr << "ERROR: 2 DID NOT Refract ray i=" << i << "\n";
    }

    VecCopy( output_ray.P, initial_ray.P);
    VecCopy( output_ray.D, initial_ray.D);

    if(RefractRay( &initial_ray, &output_ray, n_3, n_4, -50.0, 150.0, lens_surface2)){
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";
      //      cerr << "Refracted ray i=" << i << "\n";
    } else {
                  cerr << "ERROR: 3 DID NOT Refract ray i=" << i << "\n";
    }

    VecCopy( output_ray.P, initial_ray.P);
    VecCopy( output_ray.D, initial_ray.D);

    if(RefractRay( &initial_ray, &output_ray, n_4, n_5, -50.0, 50.0, glass2)){
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";
      //      cerr << "Refracted ray i=" << i << "\n";
    } else {
                  cerr << "ERROR: 4 DID NOT Refract ray i=" << i << "\n";
    }

    VecCopy( output_ray.P, initial_ray.P);
    VecCopy( output_ray.D, initial_ray.D);

    if(RefractRay( &initial_ray, &output_ray, n_5, n_5, -5.0, 5000.0, image_plane)){
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";
      //      cerr << "Refracted ray i=" << i << "\n";
    } else {
                  cerr << "ERROR: 5 DID NOT Refract ray i=" << i << "\n";
    }



  }
}
