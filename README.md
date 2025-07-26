# BPMN + RAGAs: RDF vs. XML evaluation suite

## Quick start

Before launching any experiments, please follow all steps in:

> `Instructions.txt`

This file covers prerequisites, environment setup and OpenAI's API key management.

---

## Contents

- `BPMN-XML_context/`  
  BPMN processes serialized in XML.

- `RDF_context/`  
  Equivalent BPMN models serialized as RDF.

- `RDF-process_RAGAs.py`  
  RAGAs evaluation for the RDF-BPMN experiments.

- `XML-process_RAG-pipeline+RAGAs.py`  
  RAG pipeline for XML-BPMN processes: retrieval, query, LLM response + and RAGAs evaluation.

- `docker-compose.yml`  
  Configuration file to run Weaviate and dependencies with Docker.

- `requirements.txt`  
  Python dependencies for scripts and experiments.
