from langchain_community.document_loaders import TextLoader

# Load the RDF data as context
# loader = TextLoader("rdfSer-6BEN-enumerate all tasks.txt")
loader = TextLoader("rdfSer-6BEN-can the tasks be executed simultanously.txt") # Used for question "What happens if the user credentials are not valid?", as well
# loader = TextLoader("rdfSer-6BEN-cost and execution time.txt")
# loader = TextLoader("rdfSer-a.txt")
# loader = TextLoader("rdfSer-b.txt")
# loader = TextLoader("rdfSer-c.txt")
# loader = TextLoader("rdfSer-d.txt")
# loader = TextLoader("rdfSer-e.txt")
# loader = TextLoader("rdfSer-f.txt")
# loader = TextLoader("rdfSer-g.txt")
# loader = TextLoader("rdfSer-h.txt")
# loader = TextLoader("rdfSer-i.txt")
# loader = TextLoader("rdfSer-j.txt")
# loader = TextLoader("rdfSer-k.txt")
# loader = TextLoader("rdfSer-l.txt")
# loader = TextLoader("rdfSer-m.txt")
# loader = TextLoader("rdfSer-m-second.txt")
documents = loader.load()

# Extract and clean the full text content
full_context = " ".join([doc.page_content.strip() for doc in documents])  # Strip leading/trailing spaces
full_context = full_context.replace("\n", " ")  # Replace all newlines with single spaces
full_context = " ".join(full_context.split())  # Normalize multiple spaces into single spaces

# Define evaluation data
questions = [
    # "Enumerate all tasks and all subprocesses inside 'User coordinated bot' pool, subsequent to the task labeled 'Add product to cart'."
    # "Can the tasks 'Notify user of failed authentication' and 'Provide delivery & invoicing data' be executed simultaneously and why or why not?"
    # "What happens if the user credentials are not valid?"
    "What happens inside the corresponding subprocess if the user credentials are not valid?"
    # "What are the cost and execution time for every task within the bot authentication microflow?"
    #"Identify the tasks in the order described in the process."
    #"Name all the tasks after the event X."
    #"Identify all the events before the event Z, in the order executed in the process, and their trigger (Cancel, Compensation, Conditional, Error, Escalation, Link, Message, Signal, Timer)"
    #"Can the task Y be executed if the task A was not previously executed? Why or why not?"
    #"Which are the tasks that are executed until the end of the process if condition v3 is true?"
    #"Can the tasks A and B be executed simultaneously? Why or why not?"
    #"How does the boundary message event m affect the process?"
    #"What triggers the compensation flow in the current process?"
    #"After the task B is executed, what conditions need to be met for task C to be executed?"
    #"How does the participant P1 interact with other participants of the process?"
    #'Which participant of the process sends the first message?'
    #"Identify all events before the event E3, in the order executed in the process."
    #'What does the start of the process executed by P2 depend on?'
    #"Inside which lane is the task D? Which task comes before the task D?"
    #"What is the execution of task C conditioned by?"
    #"Which data objects are used as outputs from the task A and where are these data objects used as input?"
]

ground_truths = [
    # "All tasks and all subprocesses inside 'User coordinated bot' pool, subsequent to the task labeled 'Add product to cart' are: 'Edit quantity' (belonging to the subprocess labeled 'Repeat for each desired product'), 'Bot authentication microflow' (subprocess), 'Provide delivery & invoicing data', 'Notify user of failed authentication', 'Request billing', 'Trigger payment with saved payment method', 'Approve online payment', 'Look for email from merchant', 'Save invoice', 'Generate order confirmation SMS', 'Generate failure SMS', 'Track order', 'Leave feedback', 'Notify delay'."
    # "No, the tasks 'Notify user of failed authentication' and 'Provide delivery & invoicing data' cannot be executed simultaneously because they follow an exclusive gateway labeled 'Authentication successful?', allowing only one path to be executed. The process either handles a failed authentication or proceeds with a successful authentication, but not both."
    "If the user credentials are not valid, the system checks for multiple failed attempts. If there are no more attempts, the account is locked, the failed attempts counter is incremented, the user is notified of the account status and the subprocess ends with a failed authentication. If the attempts are under limit, the user is granted a grace period for one more try before lockout and if the user accepts the additional authentication attempt, the system returns to the earlier task of retrieving user credentials."
    # "The cost and execution time for every task within the bot authentication microflow are: a cost of 0.01 and execution time of 2 seconds for 'Retrieve user credentials', a cost of 0.02 and execution time of 5 seconds for 'Validate credentials', a cost of 0.03 and execution time of 6 seconds for 'Check access rights' a cost of 0.01 and execution time of 1 second for 'Create authentication log', a cost of 0.01 and execution time of 1 second for 'Check for multiple failed attempts', a cost of 0.03 and execution time of 5 seconds for 'Lock account', a cost of 0.01 and execution time of 1 second for 'Increment failed attempts counter', a cost of 0.02 and execution time of 2 seconds for 'Notify user of account status'."
    #"The tasks in the order described in the process are: A, B, C."
    #"The tasks after the event X are B and C."
    #"The events before the event Z, in the order executed in the process, are: Start event, X and Y. The trigger for Start is not specified. The trigger for X is a message. The trigger for Y is also a message."
    #"The task Y cannot be executed if the task A was not previously executed, because the parallel non-exclusive gateway converges all the incoming sequence flows from the tasks A, B and C before executing the task Y."
    #"The tasks that are executed until the end of the process if condition v3 is true are C and Y."
    #"No, the tasks A and B cannot be executed simultaneously because at the exclusive event-based gateway labeled 'e?', the decision is based on which of the succeeding intermediate event (e1, e2 or e3) occurs first and only one of the paths can be taken."
    #"If the boundary message event m arrives during the execution of the task B, this task is interrupted and task D will be executed instead. After task D is completed, the process ends at the subsequent End event."
    #"The compensation flow in the current process is triggered by the compensation end event e."
    #"After the task B is executed, the execution of task C is conditioned by the arrival of the message M2 produced by the task F after its execution."
    #"The participant P1 interacts with other participants of the process by sending the message M1 from the task B to the participant P2 and receiving the message M2 from the participant P2 to the task D."
    #'The participant “P1” sends the first message “M1” in the process.'
    #"The events before the event E3, in the order executed in the process, are Start event, E1 and E2."
    #"The start of the process executed by P2 depends on the arrival of message M1 sent by P1 from E1."
    #"The task D is inside the lane L2. The task that comes before the task D is the task C."
    #"The execution of task C is conditioned by the execution and completion of task B and by data object d1 as input for task C."
    #"The data objects used as outputs from the task A are d1 and d2. The data object d1 is used as input to the task C and the data object d2 is used as input to the task D."
]

answers = [
    # "The tasks and subprocesses inside 'User coordinated bot' pool, subsequent to the task labeled 'Add product to cart' are the following: 'Edit quantity', 'Bot authentication microflow', 'Authenticate user of failed authentication', 'Provide delivery & invoicing data', 'Request billing', 'Trigger payment with saved payment method', 'Approve online payment', 'Look for email from merchant', 'Save invoice', 'Generate order confirmation SMS', 'Generate failure SMS', 'Track order', 'Notify delay', and 'Leave feedback'."
    # "Based on the model, the tasks 'Notify user of failed authentication' and 'Provide delivery & invoicing data' cannot be executed simultaneously. This is because these tasks are conditioned by different outcomes of the 'Authentication_successful?' gateway. The task 'Notify user of failed authentication' is triggered when authentication is unsuccessful, whereas the task 'Provide delivery & invoicing data' is triggered when authentication is successful. Thus, these tasks target opposing scenarios and cannot occur at the same time."
    'If the user credentials are not valid, the subprocess in this context diverges at the Exclusive Gateway "Credentials valid?". It will then execute the Task "Check for multiple failed attempts", followed by another decision at "Attempts under limit?". If the number of attempts is under the limit a grace period would be given for another attempt otherwise, the "Lock account" task runs, leading to "Increment failed attempts counter", and "Notify user of account status" tasks. A notification is then sent out, marking the end of the flow with the End Event "Failed authentication".'
    # "The cost and execution time for each task within the bot authentication microflow are as follows: 'Retrieve user credentials' costs 0.01 and takes 2 seconds, 'Validate credentials' costs 0.02 and takes 5 seconds, 'Check access rights' costs 0.03 and takes 6 seconds, 'Create authentication log' costs 0.01 and takes 1 second, 'Check for multiple failed attempts' costs 0.01 and takes 1 second, 'Lock account' costs 0.03 and takes 5 seconds, 'Increment failed attempts counter' costs 0.01 and takes 1 second, and 'Notify user of account status' costs 0.02 and takes 2 seconds."
    #"The tasks in the order described in the process are A, B, and C."
    #"The tasks after the event X are Task B and Task C."
    #'The events before the event Z are the "Start Event (BPMN)-35353", event X, and event Y. The Start Event (BPMN)-35353 does not have any particular triggers, event X is triggered by a message, and event Y is also triggered by a message.'
    #"No, the task Y cannot be executed if the task A was not previously executed. This is because, according to the information provided, task Y comes after a Non-exclusive Gateway (converging), and task A is one of the tasks that lead up to this gateway. This means that task A must be completed before the process can continue and reach task Y."
    #'If condition v3 is true, then the tasks executed until the end of the process are "C" and "Y".'
    #"No, the tasks A and B cannot be executed simultaneously. The diagram describes a workflow where task B can only follow after the completion of an intermediate event (e2), which is one of the possible next steps after an exclusive event-based gateway (e?). Task A also follows after a different intermediate event (e1), which is another possible outcome of the same gateway. Since an exclusive gateway only allows for one sequence to follow, tasks A and B cannot be executed at the same time."
    #'The boundary message event "m" in this process acts as an interrupting event. It disrupts the normal flow of the process and directs it towards another process step. In this case, after the event "m" happens, the process proceeds to Task D instead of continuing its normal sequence.'
    #'The compensation flow in the current process is triggered by the End Event labelled as "e", which has its compensation attribute set to "Yes".'
    #"The execution of task C requires the completion of task F as depicted by the message flow M2 in this model. After the completion of task B, task F should be executed for task C to take place."
    #'The participant P1 interacts with other participants through tasks and message flows. P1 initiates a message flow "M1" from task "B" to participant P2. Later, P2 sends back an interaction "M2" to task "D". Therefore, participant P1 is able to both send and receive interactions within the process.'
    #"The events executed before the event E3, in order, are: Start Event (BPMN)-35079, E1, and E2."
    #"The start of the process executed by P2 depends on the Intermediate Event E1 from the process executed by P1. This is indicated by the message flow M1 which runs from E1 to the start event E2 in P2."
    #'The task D is inside the lane labeled "L2". The task that comes before task D is task C.'
    #"The execution of task C is conditioned by the completion of task B and the availability of data object d1, as indicated by the 'Subsequent' relationship from task B to task C and the 'Data Association' from data object d1 to task C."
    #'The data objects used as outputs from Task A are "d1" and "d2". The data object "d1" is used as input for Task C, and the data object "d2" is used as input for Task D.'
]

# Use the cleaned full context
contexts = [[full_context] for _ in questions]  # Every question gets the same cleaned RDF file as context

# Define data dictionary
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,  # Use cleaned full RDF file as context
    "ground_truths": ground_truths,
    "reference": ground_truths,
}

# Prepare the evaluation dataset
from datasets import Dataset
dataset = Dataset.from_dict(data)

# Define evaluation metrics
from ragas import evaluate
from ragas.metrics import (
    ResponseRelevancy,
    SemanticSimilarity
)
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.embeddings import LangchainEmbeddingsWrapper

# Load OpenAI API key
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(model_name="gpt-4", temperature=1)
from ragas.llms import LangchainLLMWrapper
evaluator_llm = LangchainLLMWrapper(llm)

embeddings = OpenAIEmbeddings()

import numpy as np
# Computes faithfulness which measures how factually consistent the answer is with the retrieved context (XML) using OpenAI's embedding model
def semantic_faithfulness(answer, contexts):
    # If there is no retrieved context, return 0
    if not contexts or not answer:
        return 0

    # Generate embeddings for answer and retrieved contexts
    answer_embedding = embeddings.embed_query(answer)
    context_embeddings = [embeddings.embed_query(ctx) for ctx in contexts]

    # Compute cosine similarity
    similarities = [
        np.dot(answer_embedding, ctx_emb) / (np.linalg.norm(answer_embedding) * np.linalg.norm(ctx_emb))
        for ctx_emb in context_embeddings
    ]

    # Return the average faithfulness, rounded to two decimals
    return round(np.mean(similarities), 2)  # Return the average similarity score

# Compute modified faithfulness (semantic faithfulness) for each answer
semantic_faithfulness_score = [
    semantic_faithfulness(data["answer"][i], data["contexts"][i]) for i in range(len(dataset))
]

# Define the RAGAs metrics
response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=embeddings)
factual_correctness = FactualCorrectness(llm=evaluator_llm, mode="precision", atomicity="low", coverage="high")
semantic_similarity = SemanticSimilarity(embeddings=LangchainEmbeddingsWrapper(embeddings))

# Evaluate the responses
try:
    print("Evaluating answers...\n")
    result = evaluate(
        dataset=dataset, 
        metrics=[
            response_relevancy,
            factual_correctness,
            semantic_similarity
        ]
    )

    df = result.to_pandas().round(2)
    # Append faithfulness scores
    df["semantic_faithfulness"] = semantic_faithfulness_score

    import json
    # Convert DataFrame to dictionary
    data_dict = df.to_dict(orient="records")

    # Save JSON manually to prevent excessive escaping
    with open("df_RDF.json", "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)  # Prevent escaping

    print("\nDataFrames successfully exported to JSON.")

except Exception as e:
    print(f"Error during evaluation: {e}")