//
//  main.cpp
//  ParallelBlur
//
//  Created by Ran Tian on 6/1/14.
//  Copyright (c) 2014 Ran Tian. All rights reserved.
//
#define _USE_MATH_DEFINES

#define MATRIX_POWER 100

#include <cstring>
#include <cmath>
#include <cstdint>
#include <iostream>
using namespace std;

struct pixel {
	double red;
	double green;
	double blue;
	
	pixel(double r, double g, double b) : red(r), green(g), blue(b) {};
};

/*
 * The Prewitt kernels can be applied after a blur to help highlight edges
 * The input image must be gray scale/intensities:
 *     double intensity = (in[in_offset].red + in[in_offset].green + in[in_offset].blue)/3.0;
 * Each kernel must be applied to the blured images separately and then composed:
 *     blurred[i] with prewittX -> Xedges[i]
 *     blurred[i] with prewittY -> Yedges[i]
 *     outIntensity[i] = sqrt(Xedges[i]*Xedges[i] + Yedges[i]*Yedges[i])
 * To turn the out intensity to an out color set each color to the intensity
 *     out[i].red = outIntensity[i]
 *     out[i].green = outIntensity[i]
 *     out[i].blue = outIntensity[i]
 *
 * For more on the Prewitt kernels and edge detection:
 *     http://en.wikipedia.org/wiki/Prewitt_operator
 */
void prewittX_kernel(const int rows, const int cols, double * const kernel) {
	if(rows != 3 || cols !=3) {
		std::cerr << "Bad Prewitt kernel matrix\n";
		return;
	}
	for(int i=0;i<3;i++) {
		kernel[0 + (i*rows)] = -1.0;
		kernel[1 + (i*rows)] = 0.0;
		kernel[2 + (i*rows)] = 1.0;
	}
}

void prewittY_kernel(const int rows, const int cols, double * const kernel) {
    if(rows != 3 || cols !=3) {
        std::cerr << "Bad Prewitt kernel matrix\n";
        return;
    }
    for(int i=0;i<3;i++) {
        kernel[i + (0*rows)] = 1.0;
        kernel[i + (1*rows)] = 0.0;
        kernel[i + (2*rows)] = -1.0;
    }
}

/*
 * The gaussian kernel provides a stencil for blurring images based on a
 * normal distribution
 */
void gaussian_kernel(const int rows, const int cols, const double stddev, double * const kernel) {
	const double denom = 2.0 * stddev * stddev;
	const double g_denom = M_PI * denom;
	const double g_denom_recip = (1.0/g_denom);
	double sum = 0.0;
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			const double row_dist = i - (rows/2);
			const double col_dist = j - (cols/2);
			const double dist_sq = (row_dist * row_dist) + (col_dist * col_dist);
			const double value = g_denom_recip * exp((-dist_sq)/denom);
			kernel[i + (j*rows)] = value;
			sum += value;
		}
	}
	// Normalize
	const double recip_sum = 1.0 / sum;
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			kernel[i + (j*rows)] *= recip_sum;
		}
	}
}

void apply_stencil(const int radius, const double stddev, const int rows, const int cols, float * const in, float * const out) {
	const int dim = radius*2+1;
	double kernel[dim*dim];
    	gaussian_kernel(dim, dim, stddev, kernel);
    
    
    
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			const int out_offset = i + (j*rows);
			// For each pixel, do the stencil
			for(int x = i - radius, kx = 0; x <= i + radius; ++x, ++kx) {
				for(int y = j - radius, ky = 0; y <= j + radius; ++y, ++ky) {
					if(x >= 0 && x < rows && y >= 0 && y < cols) {
						const int in_offset = x + (y*rows);
						const int k_offset = kx + (ky*dim);
						out[out_offset]   += kernel[k_offset] * in[in_offset];
                                                //cout<<out[out_offset]<<" ";
					}
				}
			}
		}
	}
}
void apply_kernelY(const int radius, const int rows, const int cols, float * const in, float * const out) {
	const int dim = radius*2+1;
	double kernel[dim*dim];
	prewittY_kernel(dim, dim, kernel);
    
	for(int i = 0; i < rows; ++i) {
        
		for(int j = 0; j < cols; ++j) {
			const int out_offset = i + (j*rows);
			// For each pixel, do the stencil
            
			for(int x = i - radius, kx = 0; x <= i + radius; ++x, ++kx) {
                
				for(int y = j - radius, ky = 0; y <= j + radius; ++y, ++ky) {
					if(x >= 0 && x < rows && y >= 0 && y < cols) {
						const int in_offset = x + (y*rows);
						const int k_offset = kx + (ky*dim);
						float intensity = in[in_offset];
						out[out_offset]   += kernel[k_offset] * intensity;
					}
				}
			}
		}
	}
}

void apply_kernelX(const int radius, const int rows, const int cols, float * const in, float * const out) {
	const int dim = radius*2+1;
	double kernel[dim*dim];
	prewittX_kernel(dim, dim, kernel);
    
	for(int i = 0; i < rows; ++i) {
        
		for(int j = 0; j < cols; ++j) {
			const int out_offset = i + (j*rows);
			// For each pixel, do the stencil
            
			for(int x = i - radius, kx = 0; x <= i + radius; ++x, ++kx) {
                
				for(int y = j - radius, ky = 0; y <= j + radius; ++y, ++ky) {
					if(x >= 0 && x < rows && y >= 0 && y < cols) {
						const int in_offset = x + (y*rows);
						const int k_offset = kx + (ky*dim);
						float intensity = in[in_offset];
						out[out_offset] += kernel[k_offset] * intensity;
					}
				}
			}
		}
	}
}


void apply_geoMean(const int rows, const int cols, float * const in1, float * const in2, float * const out) {
    
    
	for(int i = 0; i < rows*cols; ++i) {
        // For each pixel, do the stencil
        out[i]   =  sqrt( in1[i]*in1[i] + in2[i] * in2[i] );
        
	}
	
}

int main( int argc, char* argv[] ) {
    
	if(argc != 3) {
		std::cerr << "Usage: " << argv[0] << " inputfile outputfile"<<std::endl;
		return 1;
	}
    
    FILE * inputfile;
    inputfile = fopen(argv[1],"r");
    FILE * outputfile;
    outputfile = fopen(argv[2], "w");
    if (inputfile != NULL && outputfile != NULL)
    {
        float image[MATRIX_POWER*MATRIX_POWER];
        char labelstring[256];
        while (!feof(inputfile))
        {
            memset(image, 0, MATRIX_POWER*MATRIX_POWER*sizeof(float));
            memset(labelstring,0,256*sizeof(char));
            for (int i = 0; i < MATRIX_POWER*MATRIX_POWER; i++)
            {
                fscanf(inputfile, "%f\t",&image[i]);
            }
            fscanf(inputfile, "%s\n",labelstring);
            
            // Create output arra
            float * outPixels = (float *) malloc(MATRIX_POWER * MATRIX_POWER * sizeof(float));
            float * outPixelsTemp1 = (float *) malloc(MATRIX_POWER * MATRIX_POWER * sizeof(float));
            float * outPixelsTemp2 = (float *) malloc(MATRIX_POWER * MATRIX_POWER * sizeof(float));
            for(int i = 0; i < MATRIX_POWER * MATRIX_POWER; ++i) {
                outPixels[i] = 0.0;
            }
            
            apply_stencil(3, 32.0, MATRIX_POWER, MATRIX_POWER, image, outPixels);
            /* 
            for (int i = 0; i < MATRIX_POWER*MATRIX_POWER; i++)
            {
                fprintf(outputfile, "%f\t",outPixels[i]);
            }
            fprintf(outputfile,"%s\n",labelstring);
            */
            // Do the naiive kernels.
            apply_kernelY(1, MATRIX_POWER, MATRIX_POWER, outPixels, outPixelsTemp1);
            apply_kernelX(1, MATRIX_POWER, MATRIX_POWER, outPixels, outPixelsTemp2);
            
            apply_geoMean(MATRIX_POWER, MATRIX_POWER, outPixelsTemp1, outPixelsTemp2, outPixels);
            //	apply_stencil(3, 32.0, rows, cols, outPixelsTemp, outPixels);

            
            for (int i = 0; i < MATRIX_POWER*MATRIX_POWER; i++)
            {
                fprintf(outputfile, "%f\t",outPixels[i]);
            }
            fprintf(outputfile,"%s\n",labelstring);
              
      }
    }
    

	return 0;
}


