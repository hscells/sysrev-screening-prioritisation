#!/usr/bin/env bash

# Comment this line out if your JAVA_HOME is somewhere else.
export JAVA_HOME=$(/usr/libexec/java_home)
plugin_path=~/elasticsearch/elasticsearch-5.3.0/bin/elasticsearch-plugin

# Build the plugin
cd ltr-query
./gradlew run#installLtrQueryPlugin

# Copy it into the elasticsearch plugin folder. This requires `elasticsearch-plugin` to be on your PATH.
${plugin_path} -v install file://$(pwd)/build/distributions/ltr-query-0.1.1-es5.3.0.zip
cd ../
