# ========================================================================
# IMPORTANT:
# First, check "Instructions.txt" for the prerequisites for this experimental setup.
# Otherwise, here's a shortcut:
#
# Run the following Docker command into the terminal, where:
# REST API (port 8080): used for standard HTTP requests like adding data, querying and managing schemas.
# gRPC API (port 50051): used for high-performance operations or streaming capabilities in specific applications.
# Docker command:
# docker run -d --name weaviate `
#  -p 8080:8080 `
#  -p 50051:50051 `
#  semitechnologies/weaviate:latest
#
# Check Weaviate container status:
# docker ps
#
# Each TextLoader line below is intended for a specific evaluation scenario.
# Therefore, each file loaded here corresponds to:
#   - A unique question in the "questions" list;
#   - A unique ground truth in the "ground_truths" list.
#
# You can test more than one question for a given BPMN XML serialized process. However, ensure that each question has its corresponding ground truth.
#
# To run the evaluation for a specific file:
#   1. Uncomment the loader line for the file you want to use.
#   2. Make sure to uncomment (or update) the corresponding question and ground truth in their respective lists.
#   3. Comment out the loader lines (and the associated question and ground truth entries) for all other files.
# ========================================================================


# 1.
from langchain_community.document_loaders import TextLoader
# Load the BPMN file
# loader = TextLoader("BPMN-XML_context/6B.EN BPMN - supervised bot.bpmn")
# loader = TextLoader("BPMN-XML_context/a.simpletasksequence.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/b.taskeventsequence.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/c.tasklesssequence.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/d.parallellbranching.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/e.decisionbranching.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/f.eventbranching.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/g.boundarybranching.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/h.compensationbranching.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/i.taskcollaboration.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/j.collapsedpoolcollaboration.anonim.bpmn")
loader = TextLoader("BPMN-XML_context/k.eventcollaboration.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/l.lanecoordination.anonim.bpmn")
# loader = TextLoader("BPMN-XML_context/m.lanecoordinationwithdata.anonim.bpmn")

# Load the data from the specified file
documents = loader.load()


# 2.
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Split the BPMN document into chunks that are easier for GPT to "digest"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=800)
chunks = text_splitter.split_documents(documents)


# 3.
from langchain_openai import OpenAIEmbeddings
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
# Generate the vector embeddings for each chunk with the OpenAI embedding model and store them in the vector database
embeddings = OpenAIEmbeddings()

# Load the OpenAI API key from .env file
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Connect to the local instance of Weaviate (the vector db)
weaviate_client = weaviate.connect_to_local(
    skip_init_checks=True
)
if weaviate_client.is_ready():
    print("Weaviate is ready!")
else:
    print("Weaviate is not ready. Make sure Docker is running.")

# Populate vector database
db = WeaviateVectorStore.from_documents(chunks, embeddings, client=weaviate_client)
retriever = db.as_retriever()


# 4. 
import re
# Function that removes excessive whitespace and newlines (that might negatively impact future evaluations), standardizes spacing
def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""

    # Normalize spaces: replace multiple spaces, tabs or newlines with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove spaces before punctuation (ensures " ;" -> ";" and " ." -> ".")
    text = re.sub(r'\s+([;,.])', r'\1', text)

    # Ensure no leading/trailing spaces
    return text.strip()


# 5.
import numpy as np
# Computes faithfulness which measures how factually consistent the answer is with the retrieved context (XML) using OpenAI's embedding model
def semantic_faithfulness(answer, contexts):
    # If there is no retrieved context, return 0
    if not contexts or not answer:
        return 0 

    # Generate embeddings for answer and retrieved contexts
    answer_embedding = embeddings.embed_query(clean_text(answer))
    context_embeddings = [embeddings.embed_query(clean_text(ctx)) for ctx in contexts]

    # Compute cosine similarity
    similarities = [np.dot(answer_embedding, ctx_emb) / (np.linalg.norm(answer_embedding) * np.linalg.norm(ctx_emb))
                    for ctx_emb in context_embeddings]

    # Return the average faithfulness, rounded to two decimals
    return round(np.mean(similarities), 2)


# 6.
from langchain.prompts import ChatPromptTemplate
# Prompt template
template = """
You are a question-answering assistant.
Keep your answer brief and direct, within 1 short paragraph.
Do not use bullet points.
Do not include additional and redundant explanations in your answer unless explicitly requested.
Always use the provided context to answer the question.
If you do not know the answer to the following question, just say 'I do not know.' or 'The provided context does not contain enough information.' and nothing else.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)


# 7.
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
# Build a chain for the RAG pipeline, chaining together the retriever, the prompt template and the LLM
# Define LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=1)

# Setup RAG pipeline
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)

# Preparing the evaluation data
# IMPORTANT: Each question in the "questions" list should correspond to the BPMN XML file loaded above.
# Similarly, each ground truth in "ground_truths" must match the evaluation scenario of the selected BPMN XML file.
questions = [
    # "Enumerate all tasks and all subprocesses inside 'User coordinated bot' pool, subsequent to the task labeled 'Add product to cart'.",
    # "Can the tasks 'Notify user of failed authentication' and 'Provide delivery & invoicing data' be executed simultaneously and why or why not?"
    # "What happens inside the corresponding subprocess if the user credentials are not valid?"
    # "What are the cost and execution time for every task within the bot authentication microflow?"
    #"Identify the tasks in the order described by the process."
    #"Name all the tasks after the event X."
    #"Identify all the events before the event Z, in the order executed in the process, and their trigger (Cancel, Compensation, Conditional, Error, Escalation, Link, Message, Signal, Timer)"
    #"Can the task Y be executed if the task A was not previously executed? Why or why not?"
    #"Which are the tasks that are executed until the end of the process if condition v3 is true?"
    #"Can the tasks A and B be executed simultaneously? Why or why not?"
    #"How does the boundary message event m affect the process?"
    #"What triggers the compensation flow in the current process?"
    #"After the task B is executed, what conditions need to be met for task C to be executed?"
    #"How does the participant P1 interact with other participants of the process?"
    #"Which participant of the process sends the first message?"
    #"Identify all events before the event E3, in the order executed in the process."
    "What does the start of the process executed by P2 depend on?"
    #"Inside which lane is the task D? Which task comes before the task D?"
    #"What is the execution of task C conditioned by?"
    #"Which data objects are used as outputs from the task A and where are these data objects used as input?"
]
ground_truths = [
    # ["All tasks and all subprocesses inside 'User coordinated bot' pool, subsequent to the task labeled 'Add product to cart' are: 'Edit quantity' (belonging to the subprocess labeled 'Repeat for each desired product'), 'Bot authentication microflow' (subprocess), 'Provide delivery & invoicing data', 'Notify user of failed authentication', 'Request billing', 'Trigger payment with saved payment method', 'Approve online payment', 'Look for email from merchant', 'Save invoice', 'Generate order confirmation SMS', 'Generate failure SMS', 'Track order', 'Leave feedback', 'Notify delay'."],
    # ["No, the tasks 'Notify user of failed authentication' and 'Provide delivery & invoicing data' cannot be executed simultaneously because the tasks follow an exclusive gateway labeled 'Authentication successful?', which allows only one path to be executed. The process will either handle a failed authentication or proceed with a successful authentication, but not both."]
    # ["If the user credentials are not valid, the system checks for multiple failed attempts. If there are no more attempts, the account is locked, the failed attempts counter is incremented, the user is notified of the account status and the subprocess ends with a failed authentication. If the attempts are under limit, the user is granted a grace period for one more try before lockout and if the user accepts the additional authentication attempt, the system returns to the earlier task of retrieving user credentials."]
    # ["The cost and execution time for every task within the bot authentication microflow are: a cost of 0.01 and execution time of 2 seconds for 'Retrieve user credentials', a cost of 0.02 and execution time of 5 seconds for 'Validate credentials', a cost of 0.03 and execution time of 6 seconds for 'Check access rights' a cost of 0.01 and execution time of 1 second for 'Create authentication log', a cost of 0.01 and execution time of 1 second for 'Check for multiple failed attempts', a cost of 0.03 and execution time of 5 seconds for 'Lock account', a cost of 0.01 and execution time of 1 second for 'Increment failed attempts counter', a cost of 0.02 and execution time of 2 seconds for 'Notify user of account status'."]
    #["The tasks in the order described in the process are: A, B, C."]
    #["The tasks after the event X are B and C."]
    #["The events before the event Z, in the order executed in the process, are: Start event, X and Y. The trigger for Start is not specified. The trigger for X is a message. The trigger for Y is also a message."]
    #["The task Y cannot be executed if the task A was not previously executed, because the parallel non-exclusive gateway converges all the incoming sequence flows from the tasks A, B and C before executing the task Y."]
    #["The tasks that are executed until the end of the process if condition v3 is true are C and Y."]
    #["No, the tasks A and B cannot be executed simultaneously because at the exclusive event-based gateway labeled 'e?', the decision is based on which of the succeeding intermediate event (e1, e2 or e3) occurs first and only one of the paths can be taken."]
    #["If the boundary message event m arrives during the execution of the task B, this task is interrupted and task D will be executed instead. After task D is completed, the process ends at the subsequent End event."]
    #["The compensation flow in the current process is triggered by the compensation end event e."]
    #["After the task B is executed, the execution of task C is conditioned by the arrival of the message M2 produced by the task F after its execution."]
    #["The participant P1 interacts with other participants of the process by sending the message M1 from the task B to the participant P2 and receiving the message M2 from the participant P2 to the task D."]
    #["The events before the event E3, in the order executed in the process, are Start, E1 and E2."]
    ["The start of the process executed by P2 depends on the arrival of message M1 sent by P1 from E1."]
    #["The task D is inside the lane L2. The task that comes before the task D is the task C."]
    #["The execution of task C is conditioned by the execution and completion of task B and by data object d1 as input for task C."]
    #["The data objects used as outputs from the task A are d1 and d2. The data object d1 is used as input to the task C and the data object d2 is used as input to the task D."]
]
answers = []
contexts = []

# Inference
for query in questions:
  answers.append(rag_chain.invoke(query))
  contexts.append([docs.page_content for docs in retriever.invoke(query)])

# Define data dictionary
data = {
    "question": [clean_text(q) for q in questions],
    "answer": [clean_text(ans) for ans in answers],
    "contexts": [[clean_text(ctx) for ctx in ctx_group] for ctx_group in contexts],
    "ground_truths": [[clean_text(gt) for gt in group] for group in ground_truths],
    "reference": [clean_text(group[0]) for group in ground_truths],
}

from datasets import Dataset
# Convert the data to a Hugging Face Dataset
dataset = Dataset.from_dict(data)


# 8.
from ragas import evaluate
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    ResponseRelevancy,
    SemanticSimilarity
)
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
# Evaluating the RAG application
# Wrap LLM using LangchainLLMWrapper for proper RAGAs integration
evaluator_llm = LangchainLLMWrapper(llm)

# Define evaluation metrics
response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=embeddings)
factual_correctness = FactualCorrectness(llm=evaluator_llm, mode="precision", atomicity="low", coverage="high")
semantic_similarity = SemanticSimilarity(embeddings=LangchainEmbeddingsWrapper(embeddings))

# Compute modified faithfulness (semantic faithfulness) for each answer
semantic_faithfulness_score = [
    semantic_faithfulness(data["answer"][i], data["contexts"][i]) for i in range(len(dataset))
]

try:
    # Print question and answers
    for question, answer in zip(questions, answers):
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print()

    # Evaluate the RAG-generated answer
    print("Evaluating the RAG-generated answer...\n")
    result_rag = evaluate(
        dataset=dataset, 
        metrics=[
            # context_precision,
            response_relevancy,
            factual_correctness,
            semantic_similarity
        ],
    )

    df_rag = result_rag.to_pandas().round(2)
    # Append faithfulness scores
    df_rag["semantic_faithfulness"] = semantic_faithfulness_score

    import json
    # Convert DataFrame to dictionary
    data_dict = df_rag.to_dict(orient="records")

    import os
    # Ensure the output directory, "BPMN-XML-process_Results", exists
    results_dir = "BPMN-XML-process_Results"
    os.makedirs(results_dir, exist_ok=True)

    # Save JSON output in the output directory
    with open(os.path.join(results_dir, "df_XML.json"), "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)  # Prevent escaping

    print("\nDataFrames successfully exported to JSON.")

except Exception as e:
    print(f"Error during evaluation: {e}")

finally:
    # Close the connection
    if weaviate_client:
        weaviate_client.close()