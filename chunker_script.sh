#! /bin/bash
# (dense1, pos_emb, drop1, lstm, drop2, dense2, opt, batch)
# machine 1 (main)
#time python3 chunker.py m1.h5 30 10 0.3 20 0.5 10 adam 8
#time python3 chunker.py m2.h5 30 10 0.3 20 0.5 10 adam 16
#time python3 chunker.py m3.h5 30 10 0.3 20 0.5 10 adam 32
#time python3 chunker.py m4.h5 30 10 0.3 20 0.5 10 adam 64
# machine 2 (181)
#time python3 chunker.py m1.h5 30 10 0.3 20 0.5 10 rmsprop 8
#time python3 chunker.py m2.h5 30 10 0.3 20 0.5 10 rmsprop 16
#time python3 chunker.py m3.h5 30 10 0.3 20 0.5 10 rmsprop 32
#time python3 chunker.py m4.h5 30 10 0.3 20 0.5 10 rmsprop 64
# machine 3 (182)
#time python3 chunker.py m1.h5 30 5 0.3 20 0.5 10 adam 16
#time python3 chunker.py m2.h5 30 15 0.3 20 0.5 10 adam 16
#time python3 chunker.py m3.h5 20 10 0.3 20 0.5 10 adam 16
#time python3 chunker.py m4.h5 40 10 0.3 20 0.5 10 adam 16
# machine 4 (183)
#time python3 chunker.py m1.h5 30 10 0.5 20 0.5 10 adam 16
#time python3 chunker.py m2.h5 30 10 0.7 20 0.5 10 adam 16
#time python3 chunker.py m3.h5 30 10 0.3 20 0.3 10 adam 16
#time python3 chunker.py m4.h5 30 10 0.3 20 0.7 10 adam 16
# machine 5(184)
#time python3 chunker.py m1.h5 30 10 0.3 15 0.5 10 adam 16
#time python3 chunker.py m2.h5 30 10 0.3 30 0.5 10 adam 16
#time python3 chunker.py m3.h5 30 10 0.3 20 0.5 3 adam 16
#time python3 chunker.py m4.h5 30 10 0.3 20 0.5 20 adam 16
