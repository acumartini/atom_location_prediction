#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>
using namespace std;

/*int Poisson(float mean) //Special technique required: Box-Muller method...
{
            float R;
            float sum = 0;
            int i;
            i=-1;
            float z;
            while(sum <=mean)
            {
                        R = (float)rand()/(float)(RAND_MAX+1);
                        z = -log(R);
                        sum+= z;
                        i++;
            }
            return i;

}
*/

int main(int argc,const char ** argv)
{
	if(argc != 5) cout<<"usage: "<<argv[0]<<"inputfile outputfile poisson_lambda scaling_factor"<<endl;
	//const int rounds = atoi(argv[2]);
	//string path = argv[2];
	FILE * outputfile, * inputfile;
	inputfile = fopen(argv[1],"r");
	outputfile = fopen(argv[2],"w");
	//srand(time(NULL));
	float ge = 0;
	char buffer[65] = {0};
	default_random_engine generator;
	poisson_distribution<int> distribution(atoi(argv[3]));
	float scale = atoi(argv[4]);
	while(!feof(inputfile))
	{
		memset(buffer,0,sizeof(char)*65);
		for(int j = 0; j < 10000; j++)
		{
			//memset(buffer,0,sizeof(char)*65);
			fscanf(inputfile,"%f\t",&ge);
			//ge = atof(buffer);
			//int temp = rand();
			//memcpy(&ge,&temp,sizeof(float));
			ge += ((float)distribution(generator))/scale;
			fprintf(outputfile,"%f\t",ge);
		}
		fscanf(inputfile,"%s\n",buffer);
		fprintf(outputfile,"%s\n",buffer);
	//cout<<buffer<<endl;
	}

	fclose(inputfile);
	fclose(outputfile);




}
