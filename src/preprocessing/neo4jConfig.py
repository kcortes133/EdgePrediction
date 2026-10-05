"""Neo4j connection settings, read from environment variables.

Set these before running scripts that query Neo4j, e.g.:
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=...
    export NEO4J_DB=monarch20250815
"""
import os

configDict = {
    'db': os.environ.get('NEO4J_DB', 'monarch20250815'),
    'uri': os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
    'user': os.environ.get('NEO4J_USER', 'neo4j'),
    'pwd': os.environ.get('NEO4J_PASSWORD', ''),
}
