python3 watch.py -high 210 -diam 8.0 &
python3 record.py -w 960 -h 720 -fps 40 -ex verylong -ss 35000 -ag 0 -dg 6 -md 4 -co 100 --awbgains 1.0,1.0 --max_frames 200
fg
