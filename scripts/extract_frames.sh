#!/bin/bash

indir=./dataset/videos/hdr/
outdir=./dataset/frames/hdr/

for FILE in $indir*
do
	FILENAME=$(basename $FILE)
	NAME=${FILENAME%_*}
	#echo NAME
        ffmpeg -i $FILE -r 0.5 -f image2 $outdir$NAME'_%03d.png'
done
