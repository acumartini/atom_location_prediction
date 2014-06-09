#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
using namespace std;

int main(int argc,const char ** argv)
{
	if(argc != 3) cout<<"usage: "<<argv[0]<<" outputfile #of_samples"<<endl;
	const int rounds = atoi(argv[2]);
	string path = argv[1];
	FILE * outputfile;
	outputfile = fopen(argv[1],"w");
	srand(time(NULL));
	float ge = 0;
	for(int i = 0; i < rounds; i++)
	{
		for(int j = 0; j < 10000; j++)
		{
			ge = ((float)(rand()%100000))/1000000;
			fprintf(outputfile,"%f\t",ge);
		}
		fprintf(outputfile,"#line%d\n",i);
	}
	fclose(outputfile);




}
