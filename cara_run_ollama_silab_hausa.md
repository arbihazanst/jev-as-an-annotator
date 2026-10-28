buka terminal baru ketik : ssh -N -L 11435:127.0.0.1:11435 -p 28659 rinaldi@37.60.243.124

password : XNPUrJ4JxVeu

biarkan terminal terbuka 

lalu buka terminal baru lagi, dan ketik : 

python evaluate_decision_models.py --models nimble:9b tev1:0.8b tev1:4b clef-flash:9b clef:27b --datasets sentiment_hau --transport systemone --output results/si_lab_hausa