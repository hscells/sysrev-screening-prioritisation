# Systematic Review Screening Prioritisation

_Using learning to rank._

## Setup

 - elasticsearch version 5.3.0
 - latest java version 1.8

The experiments use the learning to rank elasticsearch plugin from:
 
https://github.com/o19s/elasticsearch-learning-to-rank

To build the plugin, ensure you have the `JAVA_HOME` environment variable set, and have at least java version 1.8u40. 
On macOS, the to set `JAVA_HOME`, use `export JAVA_HOME=$(/usr/libexec/java_home)`.

Inside the `ltr-query` submodule, run:
 
```bash
./gradlew run#installLtrQueryPlugin
```

This will generate a zip file that can be installed to elasticsearch as a module. To install into elasticsearch, run:

```bash
elasticsearch-plugin install file:///$(pwd)/build/distributions/ltr-query-0.1.1-es5.3.0.zip
```

I have included a convenience scripts called `installPlugin.sh` in this directory that will go ahead and run these 
commands for you.

## Training A Model

To train a model, this project uses RankLib. I have forked the main repository from 
https://github.com/jattenberg/RankLib to let me use `gradle` as the build tool. To build ranklib to get the jar to start
training a model, inside the ranklib folder, run:
 
```bash
./gradlew build
```

This will produce a binary at `ranklib/build/libs/ranklib.jar`. This is an important path in the project.




