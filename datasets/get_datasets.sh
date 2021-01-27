#!/bin/bash 

# ICML 2021 State Representation Learning with Task-Irrelevant Factors of Variation in Robotics
# Anonymous Authors 2021

#download box datasets
wget https://kth.box.com/shared/static/i4aah3cv6fxzft8crgypdwv2943hzwkx.xz
#download shelf datasets
wget https://kth.box.com/shared/static/fe8iwuok21lk70g7n66n6u9dhzwek7mx.xz

#unpack box
tar -xf i4aah3cv6fxzft8crgypdwv2943hzwkx.xz
#unpack shelf
tar -xf fe8iwuok21lk70g7n66n6u9dhzwek7mx.xz

#delete box tar
rm i4aah3cv6fxzft8crgypdwv2943hzwkx.xz
#delete shelf tar
rm fe8iwuok21lk70g7n66n6u9dhzwek7mx.xz
