/*
Example usage of basic raytracer library with
a customized aspheric lens.
Welsey Erickson - 2014
*/

#include <iostream>
#include <cstdlib>
#include <math.h>
#include "vect.h"
//#include <cilk/cilk.h>
//#include <omp.h>

using namespace std;

// Parameters for the aspheric lenses. Units in mm.
#define p_n    1.453675 // index of refraction
#define p_a    10.0     // 10mm   
#define p_t    5.0      // 5mm
#define p_D    5.0      // 5mm
#define p_C    (-0.128)
#define p_k    (-0.824)
#define p_D0   17.705
//#define p_D2   -0.0211 // for some reason these give improper lens shapes - perhaps they were supposed to be squared and cubed?
//#define p_D4   -0.0871 // used these values below
#define p_D2   (0.0211*0.0211)
#define p_D4   (0.0871*0.0871*0.0871*0.0871)
#define p_Z(r) ((p_C*r*r)/(1.0+sqrt(1.0-(1.0+p_k)*p_C*p_C*r*r))+p_D0+p_D2*r*r+p_D4*r*r*r*r) // main aspheric lens function
#define p_CCD_Z 90.0  // 90 mm

// Parameters for ray generation, now inputs, but these
// are decent values that can be used.
//#define theta_range PI/5.0
//#define phi_range   PI/5.0
//#define dtheta (theta_range/100.0)
//#define dphi   (phi_range/100.0)

// Parameters for ccd screen
#define PIXEL_WIDTH 0.1
#define NUM_PIXELS 100
#define START_X (-(NUM_PIXELS/2)*PIXEL_WIDTH)
#define START_Y (-(NUM_PIXELS/2)*PIXEL_WIDTH)


/* START SURFACE DEFINITIONS */
//
// Definitions here are for the various surfaces. The raytracer
// is designed in a simple way such that rays will look for 
// surfaces to intersect with in order. This means that the rays
// should NOT hit surface2 before surface1, otherwise the results
// will be incorrect. This is not hard to garuntee.
//
// We have 5 surfaces:
// - front of first glass lens
// - back of first glass lens
// - front of second glass lens
// - back of second glass lens
// - CCD imaging plane.
//

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


// Cast a ray from the source location in direction theta, phi.
// Returns true if intersects, otherwise false
// Stores the final intersection in finalx,finaly
bool cast_ray(Vec source, Flt theta, Flt phi, int * finalx, int * finaly){


    Ray initial_ray;
    Ray output_ray;

    VecCopy(source, initial_ray.P);
    AngleToVec( 1.0, phi, theta, initial_ray.D);

    // How it works:
    // Each intersection calls refract ray on the input and output ray
    // giving the index of refraction of the two volumes, the surface
    // that connects them, and the range of values to search. If the
    // RefractRay returns true, then record the points, and copy the
    // output ray into the input ray and repeat. 

    // First Intersection
    if(RefractRay( &initial_ray, &output_ray, n_1, n_2, -5000.0, 5000.0, glass1)){
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";


     // Code for extending an output ray for debugging
     // cout << output_ray.P << "\n";
     // VecMult(100, output_ray.D, Tmp);
     // VecAdd(output_ray.P,Tmp,Fin);
     // cout << Fin << "\n\n";
     
    } else {
      cerr << "ERROR: 1 DID NOT Refract ray theta=" << theta << " phi=" << phi << "\n";
    }

    // Second Intersection
    VecCopy( output_ray.P, initial_ray.P);
    VecCopy( output_ray.D, initial_ray.D);

    if(RefractRay( &initial_ray, &output_ray, n_2, n_3, -5000.0, 5000.0, lens_surface1)){
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";
      //      cerr << "Refracted ray i=" << i << "\n";
    } else {
      cerr << "ERROR: 2 DID NOT Refract ray theta=" << theta << " phi=" << phi << "\n";
    }

    // Third Intersection
    VecCopy( output_ray.P, initial_ray.P);
    VecCopy( output_ray.D, initial_ray.D);

    if(RefractRay( &initial_ray, &output_ray, n_3, n_4, -50.0, 150.0, lens_surface2)){
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";
      //      cerr << "Refracted ray i=" << i << "\n";
    } else {
      cerr << "ERROR: 3 DID NOT Refract ray theta=" << theta << " phi=" << phi << "\n";
    }

    // Fourth Intersection
    VecCopy( output_ray.P, initial_ray.P);
    VecCopy( output_ray.D, initial_ray.D);

    if(RefractRay( &initial_ray, &output_ray, n_4, n_5, -50.0, 50.0, glass2)){
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";
      //      cerr << "Refracted ray i=" << i << "\n";
    } else {
      cerr << "ERROR: 4 DID NOT Refract ray theta=" << theta << " phi=" << phi << "\n";
    }

    // Fifth Intersection
    VecCopy( output_ray.P, initial_ray.P);
    VecCopy( output_ray.D, initial_ray.D);

    if(RefractRay( &initial_ray, &output_ray, n_5, n_5, -5.0, 5000.0, image_plane)){
      Vec Tmp,Fin;
      cout << initial_ray.P << "\n";
      cout << output_ray.P << "\n\n";
      //      cerr << "Refracted ray i=" << i << "\n";
    } else {
      cerr << "ERROR: 5 DID NOT Refract ray theta=" << theta << " phi=" << phi << "\n";
    }

    // Do binning into CCD pixels
    Flt outx = output_ray.P[0];
    Flt outy = output_ray.P[1];
    //    cerr << "end:" << outx << "," << outy;
    for(int i=0; i<NUM_PIXELS;i++){
      for(int j=0; j<NUM_PIXELS;j++){
	if( ((START_X + PIXEL_WIDTH*i) < outx) &&
	    ((START_Y + PIXEL_WIDTH*j) < outy) &&
	    ((START_X + PIXEL_WIDTH*(i+1)) > outx) && 
	    ((START_Y + PIXEL_WIDTH*(j+1)) > outy) ){
	  *finalx = i;
	  *finaly = j;
	  return true;
	}
      }
    }
    return false;
}



int main(int argc, char ** argv){

  // handle input and set up variables
  if(argc != 9){
    cerr << "argc=" << argc << "\n";
    cerr << "Usage: "<< argv[0] << "label startx starty startz theta_range dtheta phi_range dphi\n";
    return 1;
  }
  // for setting up files:
  //  ofstream outfile;
  //  outfile.open("name.txt");
  //  outfile.close();

  int label = atoi(argv[1]);

  Vec origin;
  origin[0]=atof(argv[2]);
  origin[1]=atof(argv[3]);
  origin[2]=atof(argv[4]);
  
  Flt theta_range = atof(argv[5]);
  Flt dtheta      = atof(argv[6]);
  Flt phi_range   = atof(argv[7]);
  Flt dphi        = atof(argv[8]);

  // setup out output array
  int output_array[NUM_PIXELS][NUM_PIXELS];
  for(int i=0; i<NUM_PIXELS; i++){
    for(int j=0; j<NUM_PIXELS; j++){
      output_array[i][j] = 0;
    }
  }

  // cast rays
  int endx, endy;
  for(Flt theta=-theta_range/(2.0); theta<theta_range/(2.0); theta+=dtheta){
    for(Flt phi=-phi_range/(2.0); phi<phi_range/(2.0); phi+=dphi){
      if(cast_ray(origin, theta, phi, &endx, &endy)){
	output_array[endx][endy]+=1;
      }
    }
  }

  // write output
  for(int i=0; i<NUM_PIXELS; i++){
    //    cerr << "\n"; // no line break for the format we're using
    for(int j=0; j<NUM_PIXELS; j++){
      // the actual data
      cerr << output_array[i][j] << "\t";
    }
  }
  // write label/class
  cerr << label;

}
