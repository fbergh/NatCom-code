#!/usr/bin/env bash
python prep.py syscalls/snd-cert -c 7 -o 0
java -jar negsel2.jar -alphabet file://syscalls/snd-cert/snd-cert.alpha -self train -n 10 -r 4 -c -l < test.normal | awk '{n+=$1}END{print n/NR}'
