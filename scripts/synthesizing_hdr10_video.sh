ffmpeg -r 24000/1001 -start_number 1 -i ./dataset/frames/hdr/%03d.png -pix_fmt yuv420p10le -vcodec libx265 -x265-params colormatrix=bt2020nc:transfer=smpte2084:colorprim=bt2020 -crf 8 -y hdr.mp4
