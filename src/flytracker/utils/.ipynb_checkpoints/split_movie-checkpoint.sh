#ffmpeg -i /Users/gert-janboth/Documents/flyTracker/data/movies/4arenas_noQR.h264 -vf select="between(n\,0\,20)" test.h264

#ffmpeg -i movie.mp4 -an -vf "select=between(n\,250\,750),setpts=PTS-STARTPTS" v_stream.webm
#ffmpeg -i /Users/gert-janboth/Documents/flyTracker/data/movies/4arenas_noQR.h264 -an -vf "select=between(n\,250\,750),setpts=PTS-STARTPTS" test.h264

ffmpeg -i /Users/gert-janboth/Documents/flyTracker/data/movies/4arenas_QR.h264 -codec copy -map 0 -f segment -segment_list out.csv -segment_frames 100 out%03d.mp4