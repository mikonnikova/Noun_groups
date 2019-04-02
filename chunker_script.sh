#! /bin/bash
# (dense1, pos_emb, drop1, lstm, drop2, dense2, act1, act2, opt, batch)

# machine 1 (main)
# adam vs rmsprop + batch size
#time python3 chunker.py m1.h5 30 10 0.3 20 0.5 10 relu relu adam 32 &
#time python3 chunker.py m2.h5 30 10 0.3 20 0.5 10 relu relu adam 64 &
#time python3 chunker.py m3.h5 30 10 0.3 20 0.5 10 relu relu rmsprop 32 &
#time python3 chunker.py m4.h5 30 10 0.3 20 0.5 10 relu relu rmsprop 64 &

# machine 2 (181)
# dense activation functions + min batch
#time python3 chunker.py m1.h5 30 10 0.3 20 0.5 10 relu tanh adam 64 &
#time python3 chunker.py m2.h5 30 10 0.3 20 0.5 10 tanh relu adam 64 &
#time python3 chunker.py m3.h5 30 10 0.3 20 0.5 10 tanh tanh adam 64 &
#time python3 chunker.py m1.h5 30 10 0.3 20 0.5 10 relu relu adam 16 &
# machine 3 (182)
# embedding size
#time python3 chunker.py m1.h5 30 5 0.3 20 0.5 10 relu relu adam 64 &
#time python3 chunker.py m2.h5 30 15 0.3 20 0.5 10 relu relu adam 64 &
#time python3 chunker.py m3.h5 20 10 0.3 20 0.5 10 relu relu adam 64 &
#time python3 chunker.py m4.h5 40 10 0.3 20 0.5 10 relu relu adam 64 &

# machine 4 (183)
# dropout rate
#time python3 chunker.py m1.h5 30 10 0.5 20 0.5 10 relu relu adam 64 &
#time python3 chunker.py m2.h5 30 10 0.7 20 0.5 10 relu relu adam 64 &
#time python3 chunker.py m3.h5 30 10 0.3 20 0.3 10 relu relu adam 64 &
#time python3 chunker.py m4.h5 30 10 0.3 20 0.7 10 relu relu adam 64 &

# machine 5 (184)
# dense layer size
#time python3 chunker.py m1.h5 30 10 0.3 15 0.5 10 relu relu adam 64 &
#time python3 chunker.py m2.h5 30 10 0.3 30 0.5 10 relu relu adam 64 &
#time python3 chunker.py m3.h5 30 10 0.3 20 0.5 3 relu relu adam 64 &
#time python3 chunker.py m4.h5 30 10 0.3 20 0.5 20 relu relu adam 64 &
