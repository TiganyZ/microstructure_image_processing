#!/usr/bin/gnuplot

set term qt size 800,600 font "Helvetica, 14"
set title  "Alpha-beta volume fraction with surface depth"

data_dir="AlphaBetaFraction_2021-11-02--15-02-54_images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold" # "AlphaBetaFraction_2021-11-02--16-50-11_images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold" # AlphaBetaFraction_2021-11-02--15-48-40_images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold #"AlphaBetaFraction_2021-11-02--15-02-54_images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold"

data_dir="AlphaBetaFraction_2021-11-04--09-37-03_images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold" #"AlphaBetaFraction_2021-11-04--09-32-19_images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold" #"AlphaBetaFraction_2021-11-03--11-46-22_images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold""

rm='_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold.dat'

FILES = system("ls -1 ".data_dir."/*.dat")
LABEL = system("ls -1 ".data_dir."/*.dat | sed  -e s~".data_dir."/~~ | sed -e s/AlphaBetaFraction_// | sed -e s/".rm."//"  )

# set term qt

plot for [i=1:words(FILES)] word(FILES,i) u 1:2 w points notitle #title word(LABEL,i)

# # FILES = system("ls -1 *.dat")
# plot for [data in FILES] data u 1:2 w p pt 1 lt rgb 'black' notitle'AlphaBetaFraction_OtsuThreshold_WhiteBackgroundRemoval_RemoveBakelite_images_2021-10-28--17-19-5