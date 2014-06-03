#include <stdio.h>
#include <string>
#include <dirent.h>

int main () {
	struct dirent *pDirent;
	DIR *pDir;
	int m = 0;
	// char *fspec = findfirst( "/" );
	// while (fspec != NULL) {
	// 	m++;
	//     fspec = findnext();
	// }

	pDir = opendir("../");
	if (pDir != NULL) {
	    while ( ( pDirent = readdir( pDir ) ) != NULL) {
	    	printf( pDirent->d_name );
	    	printf( "\n" );
	        m++;
	    }
	    closedir (pDir);
	}
	printf( "m = %d\n", m - 2 );
}