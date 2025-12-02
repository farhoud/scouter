#!/bin/bash

# Download APOC plugin for Neo4j
# Assumes Neo4j 5.x

APOC_VERSION=5.26.17
PLUGINS_DIR=./neo4j/plugins
JAR_FILE=$PLUGINS_DIR/apoc-${APOC_VERSION}-core.jar

mkdir -p $PLUGINS_DIR

# Remove any existing APOC jars to avoid conflicts
rm -f $PLUGINS_DIR/apoc*.jar

if [ -f "$JAR_FILE" ]; then
    echo "APOC $APOC_VERSION already downloaded at $JAR_FILE"
else
    echo "Downloading APOC $APOC_VERSION..."
    curl -L https://github.com/neo4j/apoc/releases/download/${APOC_VERSION}/apoc-${APOC_VERSION}-core.jar -o $JAR_FILE
    echo "APOC downloaded to $PLUGINS_DIR"
fi