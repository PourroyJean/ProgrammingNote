#!/bin/bash



for _file  in ../notebook/*.ipynb
do
   
    cp $_file `basename "$_file"`
    jupyter nbconvert `basename "$_file"` --to markdown --output-dir=./Programmation/
    rm `basename "$_file"`
done
