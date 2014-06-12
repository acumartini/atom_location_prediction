#!/bin/bash

#FILE HANDLING DEFINITIONS
outputfolder="./data/"
ext_points=".pts"
ext_data=".dat"

#MAIN GRID (determines number of classes)
gridmax="11"
pstart="-1.0"
pstep="0.2"

#SUB GRID (determines number of base training samples per class)
modgridmax="5"
pmodstart="-0.02"
pmodstep="0.01"

for (( j=0; j<$gridmax; j++ ))
do
    for (( i=0; i<$gridmax; i++ ))
    do
 	label=$(( gridmax * j + i ))
	px=$( echo "$pstart + $i * $pstep" | bc )
	py=$( echo "$pstart + $j * $pstep" | bc )
	echo "Running: ./a " $i $j $label $px $py 
	echo "$outputfolder$label$ext_points"
	echo "$outputfolder$label$ext_data"

	#first run makes the file
	./a6 $label $px $py 0 .6 .06 .6 .06 > "$outputfolder$label$ext_points" 2> "$outputfolder$label$ext_data"
	for (( k=0; k<$modgridmax; k++ ))
	do
	    for (( l=0; l<$modgridmax; l++ ))
	    do
		echo "--- Running: ./a " $i $j $label $px $py 
		pxmod=$( echo "$px + $pmodstart + $l * $pmodstep" | bc )
		pymod=$( echo "$py + $pmodstart + $k * $pmodstep" | bc )

		#hack to make line breaks in files
		echo "" >> "$outputfolder$label$ext_points"
		echo "" >> "$outputfolder$label$ext_data"
		./a6 $label $pxmod $pymod 0 .6 .06 .6 .06 >> "$outputfolder$label$ext_points" 2>> "$outputfolder$label$ext_data"
	    done
	done
    done
done
