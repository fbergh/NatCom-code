#!/usr/bin/env bash
python prep.py syscalls/snd-cert -c 20 -o 0
java -jar negsel2.jar -alphabet file://alphabet -self train -n 10 -r 4 -c -l < test.normal > results
