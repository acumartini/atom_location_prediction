#!/bin/bash

#I can export these so that I can grab them from the qsubbed env.

#FILE HANDLING DEFINITIONS
export outputfolder="/home/users/wwe/prog/data/"
export raytrace_command="/home/users/wwe/prog/a6"
export ext_points=".pts"
export ext_data=".dat"

#MAIN GRID (determines number of classes)
export gridmax="11"
export pstart="-1.0"
export pstep="0.2"

#SUB GRID (determines number of base training samples per class)
export modgridmax="5"
export pmodstart="-0.02"
export pmodstep="0.01"

#RAYTRACE PARAMETERS
export RAYZ="0"
export DTHETA="0.06"
export THETARANGE="0.6"
export DPHI="0.06"
export PHIRANGE="0.6"


for (( j=0; j<$gridmax; j++ ))
do
    for (( i=0; i<$gridmax; i++ ))
    do
 	export label=$(( gridmax * j + i ))
	export px=$( echo "$pstart + $i * $pstep" | bc )
	export py=$( echo "$pstart + $j * $pstep" | bc )
	echo "Running: ./a " $i $j $label $px $py 
	echo "$outputfolder$label$ext_points"
	echo "$outputfolder$label$ext_data"

	#first run makes the file
	$raytrace_command $label $px $py $RAYZ $THETARANGE $DTHETA $PHIRANGE $DPHI > "$outputfolder$label$ext_points" 2> "$outputfolder$label$ext_data"

	#NOW WE LAUNCH sublevels
	qsub -q batch -V -o ./logs/$label.log -e ./logs/$label.err ./runscript_qsub.pbs

    done
done
