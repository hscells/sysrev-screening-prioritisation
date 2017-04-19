#!/usr/bin/env bash

ranklib_path=./ranklib/build/libs/ranklib.jar

if [ ! -f ${ranklib_path} ]; then
    cd ./ranklib/
    ./gradlew build
    cd ../
fi

java -jar ${rank


_path} -train $1 -ranker $2 -save $3